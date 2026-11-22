function run_matrix_sl(rnd, design, part, nparts)
% Era-2 §9 matrix in Simulink (case30_pidv2.slx, fast restart).
%   rnd    'A' = no rail, no rate limit | 'B' = v1.2: |P_set - Pm0| <= 1.0, no rate limit
%   design 'pole' (seeds_v2 generalist) | 'ceilA' / 'ceilB' (gains file ceiling_<rnd>.mat)
% Scenarios: matrix_scen.mat (1,200 seed-7 eval + G1 alpha-reach grid). Writes
% matrix_<rnd>_<design>_<part>.mat with peak, t_s (1e-3 band), TV(Pm) per scenario.
here = fileparts(mfilename('fullpath'));  b30 = fullfile(here, '..', '..', 'bus_30');
addpath(b30);  L = case30_lib();
P0 = load(fullfile(b30, 'case30_simulink_data.mat'));  B = load(fullfile(b30, 'case30_base.mat'));
NG = numel(P0.H);  [Yr, Vr] = L.kron_reduce(B.Y_pre, B.gen_bus(:), P0.Xdp);
S = load(fullfile(here, 'matrix_scen.mat'));
if strcmp(design, 'pole'), g = S.Gpole; else, g = load(fullfile(here, ['ceiling_' design(end) '.mat'])).G; end
G = repmat(reshape(g, 1, NG, 3), 160, 1, 1);
if rnd == 'A', lo = -Inf(NG, 1); hi = Inf(NG, 1); else, lo = P0.Pm0(:) - 1; hi = P0.Pm0(:) + 1; end
types = {'gen', 'load', 'line'};
N = numel(S.typ);  idx = part:nparts:N;
mdl = 'case30_pidv2';  load_system(fullfile(here, [mdl '.slx']));
set_param(mdl, 'FastRestart', 'on');
peak = nan(numel(idx), 1); ts = peak; tv = peak;
t0 = tic;
for j = 1:numel(idx)
    k = idx(j);  typ = types{S.typ(k)};
    P = L.scenario_P(P0, B, Yr, Vr, typ, S.alpha(k), S.loc(k), S.tf(k), S.tc(k));  %#ok<NASGU>
    trip = zeros(NG, 1);  if strcmp(typ, 'gen'), trip(S.loc(k)) = S.alpha(k) * P0.Pm0(S.loc(k)); end
    C = struct('G', G, 'Pm0', P0.Pm0(:), 'lo', lo, 'hi', hi, 'rate', Inf, 'h', 1e-3, 'Tl', 0.05, ...
               'Nd', 50, 'trip', trip, 'freeze', double(trip > 0), 'dead', zeros(NG, 1));  %#ok<NASGU>
    o = sim(mdl, 'SrcWorkspace', 'current');
    t = o.omega.Time;  dev = max(abs(L.flat(o.omega) - 1), [], 2);  pm = L.flat(o.Pm);
    peak(j) = max(dev);
    le = find(dev > 1e-3, 1, 'last');
    if isempty(le), ts(j) = 0; else, ts(j) = max(t(le) - S.tf(k), 0); end
    tv(j) = sum(abs(diff(pm)), 'all');
end
set_param(mdl, 'FastRestart', 'off');
save(fullfile(here, sprintf('matrix_%s_%s_%d.mat', rnd, design, part)), 'idx', 'peak', 'ts', 'tv');
fprintf('%s/%s part %d: %d sims in %.0f s\n', rnd, design, part, numel(idx), toc(t0));
end
