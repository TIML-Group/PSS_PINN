%% case30_build_model.m
% Builds case30_governor.slx: N classical machines + droop governors on a
% Kron-reduced network (same structure as the 9-bus model, sizes from the data).
% Run export_case30.py first (creates case30_simulink_data.mat).

clear; clc;
P   = load('case30_simulink_data.mat');
NG  = numel(P.H);                 % machines
NB  = size(P.Vrec_pre, 1);        % buses
mdl = 'case30_governor';

% ---------- measurement-noise blocks (outputs <name>_noisy; 0 = off) ----------
noise.std.delta = 0.01;   noise.std.omega = 5e-5;  noise.std.Pm = 0.005;
noise.std.Id    = 0.005;  noise.std.Iq    = 0.005; noise.std.Vm = 0.001;
noise.std.Th    = 0.01;
noise.process_std = 0;    % random power disturbance in the swing equation [pu]
noise.seed        = 1;
noise.Ts          = 1e-3;

if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl); open_system(mdl);
blk  = @(lib, name, varargin) add_block(lib, [mdl '/' name], varargin{:});
wire = @(src, dst) add_line(mdl, src, dst, 'autorouting', 'on');
pos  = @(x, y) [x y x+60 y+40];

% ---------- time ----------
blk('simulink/Sources/Clock', 'Clock', 'Position', pos(130, 430));

% ---------- network + governor set-point (MATLAB Function) ----------
blk('simulink/User-Defined Functions/MATLAB Function', 'Network', 'Position', [250 380 400 520]);
code = sprintf([ ...
 'function [Pe, PmSet, Id, Iq, Vm, Th] = Network(delta, t, P)\n' ...
 '%% Kron-reduced network; fault / clearing / delta_p switching by time\n' ...
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
 'E   = (P.Ed + 1j*P.Eq) .* rot;\n' ...
 'I   = Yr * E;\n' ...
 'Pe  = real(E .* conj(I));\n' ...
 'Idq = I ./ rot;\n' ...
 'Id  = real(Idq);  Iq = imag(Idq);\n' ...
 'V   = Vr * E;\n' ...
 'Vm  = abs(V);     Th = angle(V);\n' ...
 'end\n']);
rt = sfroot;
ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', [mdl '/Network']);
ch.Script = code;
pd = ch.find('-isa', 'Stateflow.Data', 'Name', 'P');
pd.Scope = 'Parameter';

% ---------- states ----------
onesN = sprintf('ones(%d,1)', NG);
blk('simulink/Continuous/Integrator', 'delta', 'InitialCondition', 'P.delta0', 'Position', pos(900, 100));
blk('simulink/Continuous/Integrator', 'omega', 'InitialCondition', onesN,      'Position', pos(700, 200));
blk('simulink/Continuous/Integrator', 'Pm',    'InitialCondition', 'P.Pm0',    'Position', pos(700, 600));

blk('simulink/Sources/Constant', 'One', 'Value', onesN, 'VectorParams1D', 'off', 'Position', pos(700, 300));
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

if noise.process_std > 0
    set_param([mdl '/Swing'], 'Inputs', '+--+');
    blk('simulink/Sources/Random Number', 'ProcNoise', ...
        'Mean', sprintf('zeros(%d,1)', NG), ...
        'Variance', sprintf('%g*ones(%d,1)', noise.process_std^2, NG), ...
        'Seed', sprintf('%d+(1:%d)''', 1000 * noise.seed, NG), 'SampleTime', num2str(noise.Ts), ...
        'VectorParams1D', 'off', 'Position', pos(380, 120));
    wire('ProcNoise/1', 'Swing/4');
end

% governor: d(Pm)/dt = (PmSet - invR.*dw - Pm) / Tg   (invR = 0 for condensers)
blk('simulink/Math Operations/Gain', 'Droop', 'Gain', 'P.invR', 'Multiplication', 'Element-wise(K.*u)', 'Position', pos(480, 680));
blk('simulink/Math Operations/Sum',  'Gov',   'Inputs', '+--', 'Position', pos(560, 600));
blk('simulink/Math Operations/Gain', 'InvTg', 'Gain', '1/P.Tg', 'Position', pos(630, 600));
wire('Network/2', 'Gov/1');  wire('dw/1', 'Droop/1');  wire('Droop/1', 'Gov/2');
wire('Pm/1', 'Gov/3');       wire('Gov/1', 'InvTg/1'); wire('InvTg/1', 'Pm/1');

wire('delta/1', 'Network/1');  wire('Clock/1', 'Network/2');

% ---------- logging ----------
names = {'delta', 'omega', 'Pm', 'Id', 'Iq', 'Vm', 'Th'};
srcs  = {'delta/1', 'omega/1', 'Pm/1', 'Network/3', 'Network/4', 'Network/5', 'Network/6'};
sizes = [NG NG NG NG NG NB NB];
for k = 1:numel(names)
    tw = ['log_' names{k}];
    blk('simulink/Sinks/To Workspace', tw, 'VariableName', names{k}, ...
        'SaveFormat', 'Timeseries', 'Position', pos(1100, 60 + 90 * k));
    wire(srcs{k}, [tw '/1']);

    s = noise.std.(names{k});
    if s <= 0, continue; end
    n  = sizes(k);  y = 60 + 90 * k;
    nb = ['noise_' names{k}];  ad = ['add_' names{k}];  twn = ['log_' names{k} '_noisy'];
    blk('simulink/Sources/Random Number', nb, ...
        'Mean', sprintf('zeros(%d,1)', n), 'Variance', sprintf('%g*ones(%d,1)', s^2, n), ...
        'Seed', sprintf('%d+(1:%d)''', 100 * k * noise.seed, n), ...
        'SampleTime', num2str(noise.Ts), 'VectorParams1D', 'off', 'Position', pos(1250, y + 30));
    blk('simulink/Math Operations/Sum', ad, 'Inputs', '++', 'Position', pos(1350, y));
    blk('simulink/Sinks/To Workspace', twn, 'VariableName', [names{k} '_noisy'], ...
        'SaveFormat', 'Timeseries', 'Position', pos(1450, y));
    wire(srcs{k}, [ad '/1']);  wire([nb '/1'], [ad '/2']);  wire([ad '/1'], [twn '/1']);
end

% ---------- solver ----------
set_param(mdl, 'StopTime', '8', 'SolverType', 'Fixed-step', 'Solver', 'ode14x', ...
          'FixedStep', '1e-3', 'ReturnWorkspaceOutputs', 'on');
save_system(mdl);
fprintf('Built %s.slx (%d machines, %d buses).\n', mdl, NG, NB);
