function speed_probe_sl()
% Wiki S16: Simulink runner timing (pair: speed_probe.py). Same 64 cases (matrix_scen.mat eval
% idx 1..64), v1.3 semantics (all are alpha < 1 or load/line), rail 1.0, no rate limit, pole
% generalist, ode4 at 1 ms, 8 s. Times: model load, fast-restart first sim (compile), single
% scenario (best of 3 warm), and the 64 batch run serially with fast restart (incl. per-scenario
% parameter build, as run_v13_sl does). Also one single run WITHOUT fast restart.
% Writes speed_probe_sl.mat.
here = fileparts(mfilename('fullpath'));  b30 = fullfile(here, '..', '..', 'bus_30');
addpath(b30);  L = case30_lib();
P0 = load(fullfile(b30, 'case30_simulink_data.mat'));  B = load(fullfile(b30, 'case30_base.mat'));
NG = numel(P0.H);  [Yr, Vr] = L.kron_reduce(B.Y_pre, B.gen_bus(:), P0.Xdp);
S = load(fullfile(here, 'matrix_scen.mat'));
G = repmat(reshape(S.Gpole, 1, NG, 3), 160, 1, 1);
lo = P0.Pm0(:) - 1;  hi = P0.Pm0(:) + 1;  types = {'gen', 'load', 'line'};
mdl = 'case30_pidv2';
tic;  load_system(fullfile(here, [mdl '.slx']));  t_load = toc;
    function [P, C] = params(k)
        typ = types{S.typ(k)};
        P = L.scenario_P(P0, B, Yr, Vr, typ, S.alpha(k), S.loc(k), S.tf(k), S.tc(k));
        trip = zeros(NG, 1);  if strcmp(typ, 'gen'), trip(S.loc(k)) = S.alpha(k) * P0.Pm0(S.loc(k)); end
        C = struct('G', G, 'Pm0', P0.Pm0(:), 'lo', lo, 'hi', hi, 'rate', Inf, 'h', 1e-3, 'Tl', 0.05, ...
                   'Nd', 50, 'trip', trip, 'freeze', zeros(NG, 1), 'dead', zeros(NG, 1));
    end
    function pk = one(k)
        [P, C] = params(k);  %#ok<ASGLU>
        o = sim(mdl, 'SrcWorkspace', 'current');
        pk = max(abs(L.flat(o.omega) - 1), [], 'all');
    end
% single run, no fast restart (each sim recompiles)
set_param(mdl, 'FastRestart', 'off');
tic;  one(1);  t_single_nofr = toc;
set_param(mdl, 'FastRestart', 'on');
tic;  one(1);  t_first_fr = toc;                 % compile into fast restart
ts = zeros(3, 1);  for r = 1:3, tic; one(1); ts(r) = toc; end
t_single = min(ts);
peaks = zeros(64, 1);
tic;  for k = 1:64, peaks(k) = one(k); end;  t_batch = toc;
set_param(mdl, 'FastRestart', 'off');
save(fullfile(here, 'speed_probe_sl.mat'), 't_load', 't_single_nofr', 't_first_fr', 't_single', 't_batch', 'peaks');
fprintf('load %.1f s | single no-FR %.2f s | first FR %.2f s | single warm %.3f s | 64 batch %.1f s (%.2f cases/s)\n', ...
        t_load, t_single_nofr, t_first_fr, t_single, t_batch, 64 / t_batch);
end
