%% case30_check_model.m
% Two checks of case30_governor.slx:
%   1. no fault           -> every signal must stay flat (true equilibrium)
%   2. one example fault  -> Simulink must match the fast model used by the
%                            delta_p search (same equations, RK4)

clear; clc;
L   = case30_lib();
P0  = load('case30_simulink_data.mat');
B   = load('case30_base.mat');
mdl = 'case30_governor';
NG  = numel(P0.H);
[Yr_pre, Vr_pre] = L.kron_reduce(B.Y_pre, B.gen_bus(:), P0.Xdp);

if bdIsLoaded(mdl), close_system(mdl, 0); end
load_system(mdl);

% ---------- 1. no fault ----------
P = P0;
P.t_fault = 1e9;  P.t_clear = 1e9;
o = sim(mdl);
w = L.flat(o.omega);  d = L.flat(o.delta);
fprintf('No fault:  max|omega-1| = %.2e   max delta drift = %.2e rad\n', ...
        max(abs(w - 1), [], 'all'), max(abs(d - d(1, :)), [], 'all'));

% ---------- 2. example fault ----------
TYPE = 'load';  ALPHA = 0.5;  LOC = 7;  TF = 1.0;  TC = 2.5;   % edit to try others
P = L.scenario_P(P0, B, Yr_pre, Vr_pre, TYPE, ALPHA, LOC, TF, TC);
o = sim(mdl);
ts = o.omega.Time;  ws = L.flat(o.omega);
[tf_, wf] = L.fast_traj(P, zeros(NG, 1), 1e9, 1e9, 1e-3, 8);
wf_i = interp1(tf_, wf, ts);
fprintf('%s fault (alpha=%.2f, loc=%d): max|Simulink - fast model| = %.2e pu  (peak dev %.4f)\n', ...
        TYPE, ALPHA, LOC, max(abs(ws - wf_i), [], 'all'), max(abs(ws - 1), [], 'all'));

figure('Name', 'case30 model check');
subplot(2, 1, 1); plot(ts, ws, 'LineWidth', 1.2); hold on; grid on;
set(gca, 'ColorOrderIndex', 1); plot(tf_, wf, '--');
xline(TF, 'k-'); xline(TC, 'r-');
yline(1.005, 'k:'); yline(0.995, 'k:');
title(sprintf('\\omega: solid = Simulink, dashed = fast model  (%s fault)', TYPE));
legend(arrayfun(@(i) sprintf('G%d', i), 1:NG, 'uni', 0), 'Location', 'best');
subplot(2, 1, 2); plot(ts, ws - wf_i); grid on;
title('difference Simulink - fast model'); xlabel('t [s]');
