%% build_ieee9_gov_model.m
% Builds a Simulink model equivalent to cross_validation_datagen_governor.py
% (3 classical machines + droop governor, IEEE 9-bus, Kron-reduced network).
% Run export_for_simulink.py first to create ieee9_gov_simulink_data.mat.

clear; clc;
P   = load('ieee9_gov_simulink_data.mat');   % struct used by all blocks
if ~isfield(P, 'dPc')    % corrective delta_p fields (inactive by default)
    P.dPc = zeros(3, 1);  P.t_corr = 1e9;  P.t_corr_end = 1e9;
    save('ieee9_gov_simulink_data.mat', '-struct', 'P', 'dPc', 't_corr', 't_corr_end', '-append');
end
mdl = 'ieee9_governor';

% ---------- NOISE SETTINGS (standard deviations, set any to 0 to disable) ----------
% Measurement noise: added to the logged outputs only, does not change the dynamics.
noise.std.delta = 0.01;     % rad    (~0.57 deg, typical PMU angle error)
noise.std.omega = 5e-5;     % pu     (~0.003 Hz at 60 Hz)
noise.std.Pm    = 0.005;    % pu
noise.std.Id    = 0.005;    % pu
noise.std.Iq    = 0.005;    % pu
noise.std.Vm    = 0.001;    % pu     (0.1 %)
noise.std.Th    = 0.01;     % rad
% Process noise: random mechanical-power disturbance added to the swing equation [pu].
noise.process_std = 0;      % e.g. 0.01 to turn it on
noise.seed        = 1;      % change to get a different noise realisation
noise.Ts          = 0.01;   % noise sample time [s] (matches the solver step)
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl); open_system(mdl);

% ---------- helpers ----------
blk  = @(lib, name, varargin) add_block(lib, [mdl '/' name], varargin{:});
wire = @(src, dst) add_line(mdl, src, dst, 'autorouting', 'on');
pos  = @(x, y) [x y x+60 y+40];

% ---------- simulation time (fault/clearing switching is done inside Network) ----------
blk('simulink/Sources/Clock', 'Clock', 'Position', pos(130, 430));

% ---------- network + governor set-point (MATLAB Function) ----------
blk('simulink/User-Defined Functions/MATLAB Function', 'Network', 'Position', [250 380 400 520]);
code = sprintf([ ...
 'function [Pe, PmSet, Id, Iq, Vm, Th] = Network(delta, t, P)\n' ...
 '%% Kron-reduced network: replaces the Id/Iq/Vm/theta algebraic equations\n' ...
 'if t < P.t_fault - 1e-9\n' ...
 '    Yr = P.Yred_pre;   Vr = P.Vrec_pre;   PmSet = P.Pm0;\n' ...
 'elseif t < P.t_clear - 1e-9\n' ...
 '    Yr = P.Yred_fault; Vr = P.Vrec_fault; PmSet = P.Pm0 + P.dPm0;\n' ...
 'else\n' ...
 '    Yr = P.Yred_post;  Vr = P.Vrec_post;  PmSet = P.Pm0;\n' ...
 'end\n' ...
 'if t >= P.t_corr - 1e-9 && t < P.t_corr_end - 1e-9\n' ...
 '    PmSet = PmSet + P.dPc;               %% corrective delta_p\n' ...
 'end\n' ...
 'rot = exp(1j*(delta(:) - pi/2));\n' ...
 'E   = (P.Ed + 1j*P.Eq) .* rot;          %% internal EMF phasors\n' ...
 'I   = Yr * E;                            %% generator currents\n' ...
 'Pe  = real(E .* conj(I));                %% = Ed*Id + Eq*Iq\n' ...
 'Idq = I ./ rot;\n' ...
 'Id  = real(Idq);  Iq = imag(Idq);\n' ...
 'V   = Vr * E;                            %% bus voltages\n' ...
 'Vm  = abs(V);     Th = angle(V);\n' ...
 'end\n']);
rt = sfroot;
ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', [mdl '/Network']);
ch.Script = code;
pd = ch.find('-isa', 'Stateflow.Data', 'Name', 'P');
pd.Scope = 'Parameter';

% ---------- states ----------
blk('simulink/Continuous/Integrator', 'delta', 'InitialCondition', 'P.delta0',  'Position', pos(900, 100));
blk('simulink/Continuous/Integrator', 'omega', 'InitialCondition', 'ones(3,1)', 'Position', pos(700, 200));
blk('simulink/Continuous/Integrator', 'Pm',    'InitialCondition', 'P.Pm0',     'Position', pos(700, 600));

% speed deviation  dw = omega - 1
blk('simulink/Sources/Constant', 'One', 'Value', 'ones(3,1)', 'VectorParams1D', 'off', 'Position', pos(700, 300));
blk('simulink/Math Operations/Sum', 'dw', 'Inputs', '+-', 'Position', pos(800, 250));
wire('omega/1', 'dw/1');  wire('One/1', 'dw/2');

% d(delta)/dt = 2*pi*f0 * dw
blk('simulink/Math Operations/Gain', 'wBase', 'Gain', '2*pi*P.f0', 'Position', pos(820, 100));
wire('dw/1', 'wBase/1');  wire('wBase/1', 'delta/1');

% d(omega)/dt = (Pm - Pe - D*dw) / (2H)
blk('simulink/Math Operations/Gain', 'Damping', 'Gain', 'P.D', 'Multiplication', 'Element-wise(K.*u)', 'Position', pos(560, 300));
blk('simulink/Math Operations/Sum',  'Swing',   'Inputs', '+--', 'Position', pos(480, 200));
blk('simulink/Math Operations/Gain', 'Inv2H',   'Gain', '1./(2*P.H)', 'Multiplication', 'Element-wise(K.*u)', 'Position', pos(580, 200));
wire('Pm/1', 'Swing/1');  wire('Network/1', 'Swing/2');  wire('dw/1', 'Damping/1');
wire('Damping/1', 'Swing/3');  wire('Swing/1', 'Inv2H/1');  wire('Inv2H/1', 'omega/1');

% optional process noise: Pm - Pe - D*dw + w(t)
if noise.process_std > 0
    set_param([mdl '/Swing'], 'Inputs', '+--+');
    blk('simulink/Sources/Random Number', 'ProcNoise', ...
        'Mean', 'zeros(3,1)', 'Variance', sprintf('%g*ones(3,1)', noise.process_std^2), ...
        'Seed', sprintf('%d+(1:3)''', 1000*noise.seed), 'SampleTime', num2str(noise.Ts), ...
        'VectorParams1D', 'off', 'Position', pos(380, 120));
    wire('ProcNoise/1', 'Swing/4');
end

% governor: d(Pm)/dt = (PmSet - dw/R - Pm) / Tg
blk('simulink/Math Operations/Gain', 'Droop', 'Gain', '1/P.R', 'Position', pos(480, 680));
blk('simulink/Math Operations/Sum',  'Gov',   'Inputs', '+--', 'Position', pos(560, 600));
blk('simulink/Math Operations/Gain', 'InvTg', 'Gain', '1/P.Tg', 'Position', pos(630, 600));
wire('Network/2', 'Gov/1');  wire('dw/1', 'Droop/1');  wire('Droop/1', 'Gov/2');
wire('Pm/1', 'Gov/3');       wire('Gov/1', 'InvTg/1'); wire('InvTg/1', 'Pm/1');

% network inputs
wire('delta/1', 'Network/1');  wire('Clock/1', 'Network/2');

% ---------- logging ----------
names = {'delta', 'omega', 'Pm', 'Id', 'Iq', 'Vm', 'Th'};
srcs  = {'delta/1', 'omega/1', 'Pm/1', 'Network/3', 'Network/4', 'Network/5', 'Network/6'};
for k = 1:numel(names)
    tw = ['log_' names{k}];
    blk('simulink/Sinks/To Workspace', tw, 'VariableName', names{k}, ...
        'SaveFormat', 'Timeseries', 'Position', pos(1100, 60 + 90*k));
    wire(srcs{k}, [tw '/1']);
end

% ---------- measurement noise: logged as <name>_noisy ----------
sizes = [3 3 3 3 3 9 9];                       % signal widths
for k = 1:numel(names)
    s = noise.std.(names{k});
    if s <= 0, continue; end
    n  = sizes(k);
    nb = ['noise_' names{k}];
    ad = ['add_'   names{k}];
    tw = ['log_'   names{k} '_noisy'];
    y  = 60 + 90*k;
    blk('simulink/Sources/Random Number', nb, ...
        'Mean', sprintf('zeros(%d,1)', n), ...
        'Variance', sprintf('%g*ones(%d,1)', s^2, n), ...
        'Seed', sprintf('%d+(1:%d)''', 100*k*noise.seed, n), ...   % independent channels
        'SampleTime', num2str(noise.Ts), 'VectorParams1D', 'off', ...
        'Position', pos(1250, y + 30));
    blk('simulink/Math Operations/Sum', ad, 'Inputs', '++', 'Position', pos(1350, y));
    blk('simulink/Sinks/To Workspace', tw, 'VariableName', [names{k} '_noisy'], ...
        'SaveFormat', 'Timeseries', 'Position', pos(1450, y));
    wire(srcs{k}, [ad '/1']);
    wire([nb '/1'], [ad '/2']);
    wire([ad '/1'], [tw '/1']);
end

% ---------- solver: fixed-step implicit, same 0.01 s step as the Python code ----------
set_param(mdl, 'StopTime', num2str(P.t_end), 'SolverType', 'Fixed-step', ...
          'Solver', 'ode14x', 'FixedStep', '0.01', 'ReturnWorkspaceOutputs', 'on');
save_system(mdl);
fprintf('Built %s.slx. Run it with:  out = sim(''%s'');\n', mdl, mdl);
