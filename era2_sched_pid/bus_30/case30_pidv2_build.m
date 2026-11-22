%% case30_pidv2_build.m
% Builds case30_pidv2.slx: the era-2 plant (v2-nogov). Same Network/swing as
% bus_30/case30_build_model.m with the droop governor DELETED; Pm comes from a
% digital per-machine PID (Ts = 1 ms) with deviation rail, rate limit 2.5 pu/s,
% integrator clamped at the rail, ZOH gains on a global 20 Hz grid, gen trip as a
% disturbance that bypasses the actuator (tripped PID frozen). Mirrors plant_v2.py.
% Parameters: struct P (plant + scenario, as era-1 scenario_P) and struct C:
%   C.G (n_lbl x NG x 3) gains, C.lo/C.hi rail, C.rate, C.h, C.Tl, C.Nd, C.trip (NG x 1),
%   C.freeze (NG x 1) PIDs held during the fault window (Q8 legacy: trip > 0; v1.3: dead),
%   C.dead (NG x 1) v1.3 alpha = 1 disconnection: swing frozen while faulted (the runner
%   sets t_clear = 1e9 and Yred_fault = Kron reduction without the machine -> permanent).
clear; clc;
here = fileparts(mfilename('fullpath'));
P   = load(fullfile(here, '..', '..', 'bus_30', 'case30_simulink_data.mat'));
NG  = numel(P.H);
mdl = 'case30_pidv2';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
blk  = @(lib, name, varargin) add_block(lib, [mdl '/' name], varargin{:});
wire = @(a, b) add_line(mdl, a, b, 'autorouting', 'on');

blk('simulink/Sources/Clock', 'Clock');
% ---- network (continuous; switches on t like era-1) ----
blk('simulink/User-Defined Functions/MATLAB Function', 'Network');
code = sprintf([ ...
 'function [Pe, Id, Iq, Vm, Th] = Network(delta, t, P)\n' ...
 'if t < P.t_fault - 1e-9\n    Yr = P.Yred_pre;   Vr = P.Vrec_pre;\n' ...
 'elseif t < P.t_clear - 1e-9\n    Yr = P.Yred_fault; Vr = P.Vrec_fault;\n' ...
 'else\n    Yr = P.Yred_post;  Vr = P.Vrec_post;\nend\n' ...
 'rot = exp(1j*(delta(:) - pi/2));\nE = (P.Ed + 1j*P.Eq) .* rot;\nI = Yr * E;\n' ...
 'Pe = real(E .* conj(I));\nIdq = I ./ rot;\nId = real(Idq);  Iq = imag(Idq);\n' ...
 'V = Vr * E;\nVm = abs(V);  Th = angle(V);\nend\n']);
ch = sfroot().find('-isa', 'Stateflow.EMChart', 'Path', [mdl '/Network']);
ch.Script = code;
pd = ch.find('-isa', 'Stateflow.Data', 'Name', 'P');  pd.Scope = 'Parameter';
% ---- digital PID (replaces the governor) ----
blk('simulink/User-Defined Functions/MATLAB Function', 'PID');
code = sprintf([ ...
 'function [Pmech, Pm, Iint, live] = PID(omega, t, P, C)\n' ...
 'persistent I xf pm\n' ...
 'if isempty(I), I = C.Pm0; xf = zeros(size(C.Pm0)); pm = C.Pm0; end\n' ...
 'tm = t + 0.5*C.h;\n' ...
 'faulted = (tm >= P.t_fault) && (tm < P.t_clear);\n' ...
 'k = min(floor((t + 1e-9) / C.Tl) + 1, size(C.G, 1));\n' ...
 'Kp = reshape(C.G(k, :, 1), [], 1);  Ki = reshape(C.G(k, :, 2), [], 1);  Kd = reshape(C.G(k, :, 3), [], 1);\n' ...
 'e = 1 - omega(:);\n' ...
 'dterm = C.Nd * (e - xf);\n' ...
 'xf_n = xf + C.h * C.Nd * (e - xf);\n' ...
 'I_n = min(max(I + C.h * Ki .* e, C.lo), C.hi);\n' ...
 'Ps = min(max(Kp .* e + I_n + Kd .* dterm, C.lo), C.hi);\n' ...
 'pm_n = pm + min(max(Ps - pm, -C.rate * C.h), C.rate * C.h);\n' ...
 'fz = (C.freeze > 0) & faulted;\n' ...
 'xf(~fz) = xf_n(~fz);  I(~fz) = I_n(~fz);  pm(~fz) = pm_n(~fz);\n' ...
 'Pmech = pm - C.trip * double(faulted);\n' ...
 'Pm = pm;  Iint = I;  live = 1 - double((C.dead > 0) & faulted);\nend\n']);
ch = sfroot().find('-isa', 'Stateflow.EMChart', 'Path', [mdl '/PID']);
ch.Script = code;
for nm = {'P', 'C'}
    pd = ch.find('-isa', 'Stateflow.Data', 'Name', nm{1});  pd.Scope = 'Parameter';
end
set_param([mdl '/PID'], 'SystemSampleTime', '1e-3');
% ---- swing ----
blk('simulink/Continuous/Integrator', 'delta', 'InitialCondition', 'P.delta0');
blk('simulink/Continuous/Integrator', 'omega', 'InitialCondition', sprintf('ones(%d,1)', NG));
blk('simulink/Sources/Constant', 'One', 'Value', sprintf('ones(%d,1)', NG), 'VectorParams1D', 'off');
blk('simulink/Math Operations/Sum', 'dw', 'Inputs', '+-');
blk('simulink/Math Operations/Gain', 'wBase', 'Gain', '2*pi*P.f0');
blk('simulink/Math Operations/Gain', 'Damping', 'Gain', 'P.D', 'Multiplication', 'Element-wise(K.*u)');
blk('simulink/Math Operations/Sum', 'Swing', 'Inputs', '+--');
blk('simulink/Math Operations/Gain', 'Inv2H', 'Gain', '1./(2*P.H)', 'Multiplication', 'Element-wise(K.*u)');
blk('simulink/Math Operations/Product', 'LiveW');   % disconnected machine: swing frozen
blk('simulink/Math Operations/Product', 'LiveD');
wire('omega/1', 'dw/1');  wire('One/1', 'dw/2');  wire('dw/1', 'wBase/1');
wire('wBase/1', 'LiveD/1');  wire('PID/4', 'LiveD/2');  wire('LiveD/1', 'delta/1');
wire('PID/1', 'Swing/1');  wire('Network/1', 'Swing/2');  wire('dw/1', 'Damping/1');
wire('Damping/1', 'Swing/3');  wire('Swing/1', 'Inv2H/1');  wire('Inv2H/1', 'LiveW/1');  wire('PID/4', 'LiveW/2');  wire('LiveW/1', 'omega/1');
wire('delta/1', 'Network/1');  wire('Clock/1', 'Network/2');
wire('omega/1', 'PID/1');  wire('Clock/1', 'PID/2');
% ---- logging ----
names = {'delta', 'omega', 'Pm', 'Pe', 'Iint', 'Id', 'Iq', 'Vm', 'Th'};
srcs  = {'delta/1', 'omega/1', 'PID/2', 'Network/1', 'PID/3', 'Network/2', 'Network/3', 'Network/4', 'Network/5'};
for k = 1:numel(names)
    blk('simulink/Sinks/To Workspace', ['log_' names{k}], 'VariableName', names{k}, 'SaveFormat', 'Timeseries');
    wire(srcs{k}, ['log_' names{k} '/1']);
end
set_param(mdl, 'StopTime', '8', 'SolverType', 'Fixed-step', 'Solver', 'ode4', ...
          'FixedStep', '1e-3', 'ReturnWorkspaceOutputs', 'on');
save_system(mdl, fullfile(here, [mdl '.slx']));
fprintf('Built %s.slx (%d machines).\n', mdl, NG);
