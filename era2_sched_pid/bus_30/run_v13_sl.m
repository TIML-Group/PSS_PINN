function run_v13_sl(set, part, nparts, design)
% design: 'pole' (default, seeds generalist) | 'v13x' (ceiling_v13x.mat, S15 re-tune)
% Wiki S13 (v1.3, C-hybrid gen faults) screen in Simulink (case30_pidv2.slx, fast restart).
% Actuator v1.2: rail |P_set - Pm0| <= 1.0, no rate limit. Gains: pole-placement generalist.
%   set 'ev'    the 1,200 seed-7 eval scenarios of matrix_scen.mat (gen alphas all < 1: runbacks)
%       'reach' the 57-point G1 alpha grid (alpha = 1.0 points become disconnections)
%       'disc'  every gen case of 'ev' re-run at alpha = 1 (permanent disconnection)
% Gen semantics: alpha < 1 -> mechanical input -alpha*Pm0 during the fault, own PID ACTIVE;
%                alpha = 1 -> machine removed from the reduced network at t_fault, swing and PID
%                frozen, never restored.
% Writes v13sl_<set>_<part>.mat: idx, peak (all machines), peakx (faulted machine excluded
% from t_fault on), ts / tsx (last exit of the 1e-3 band after t_fault), tv (sum |dPm|).
here = fileparts(mfilename('fullpath'));  b30 = fullfile(here, '..', '..', 'bus_30');
addpath(b30);  L = case30_lib();
P0 = load(fullfile(b30, 'case30_simulink_data.mat'));  B = load(fullfile(b30, 'case30_base.mat'));
NG = numel(P0.H);  gb = B.gen_bus(:);  [Yr, Vr] = L.kron_reduce(B.Y_pre, gb, P0.Xdp);
S = load(fullfile(here, 'matrix_scen.mat'));  ne = S.n_eval;
switch set
    case 'ev',    sel = 1:ne;
    case 'reach', sel = ne+1:numel(S.typ);
    case 'disc',  sel = find(S.typ(1:ne) == 1).';
end
if nargin < 4, design = 'pole'; end
if strcmp(design, 'pole'), g = S.Gpole; else, g = load(fullfile(here, ['ceiling_' design '.mat'])).G; end
G = repmat(reshape(g, 1, NG, 3), 160, 1, 1);
lo = P0.Pm0(:) - 1;  hi = P0.Pm0(:) + 1;
types = {'gen', 'load', 'line'};
% disconnection networks: Kron reduction without machine j, embedded in NG x NG (row/col j = 0)
Yd = cell(2, 1);  Vd = cell(2, 1);
for jj = 1:2
    keep = setdiff(1:NG, jj);
    [y5, v5] = L.kron_reduce(B.Y_pre, gb(keep), P0.Xdp(keep));
    Yd{jj} = zeros(NG);  Yd{jj}(keep, keep) = y5;
    Vd{jj} = zeros(size(Vr));  Vd{jj}(:, keep) = v5;
end
idx = sel(part:nparts:numel(sel));
mdl = 'case30_pidv2';  load_system(fullfile(here, [mdl '.slx']));
set_param(mdl, 'FastRestart', 'on');
n = numel(idx);  peak = nan(n, 1);  peakx = peak;  ts = peak;  tsx = peak;  tv = peak;
t0 = tic;
for j = 1:n
    k = idx(j);  typ = types{S.typ(k)};  a = S.alpha(k);  lc = S.loc(k);
    if strcmp(set, 'disc'), a = 1; end
    P = L.scenario_P(P0, B, Yr, Vr, typ, a, lc, S.tf(k), S.tc(k));
    trip = zeros(NG, 1);  dead = zeros(NG, 1);
    if strcmp(typ, 'gen')
        if a >= 1 - 1e-12
            dead(lc) = 1;  P.Yred_fault = Yd{lc};  P.Vrec_fault = Vd{lc};  P.t_clear = 1e9;
        else
            trip(lc) = a * P0.Pm0(lc);
        end
    end
    C = struct('G', G, 'Pm0', P0.Pm0(:), 'lo', lo, 'hi', hi, 'rate', Inf, 'h', 1e-3, 'Tl', 0.05, ...
               'Nd', 50, 'trip', trip, 'freeze', dead, 'dead', dead);  %#ok<NASGU>
    o = sim(mdl, 'SrcWorkspace', 'current');
    t = o.omega.Time;  ad = abs(L.flat(o.omega) - 1);  pm = L.flat(o.Pm);
    dev = max(ad, [], 2);
    if strcmp(typ, 'gen'), ad(t > S.tf(k), lc) = 0; end
    devx = max(ad, [], 2);
    peak(j) = max(dev);  peakx(j) = max(devx);
    le = find(dev > 1e-3, 1, 'last');   if isempty(le), ts(j) = 0;  else, ts(j) = max(t(le) - S.tf(k), 0);  end
    le = find(devx > 1e-3, 1, 'last');  if isempty(le), tsx(j) = 0; else, tsx(j) = max(t(le) - S.tf(k), 0); end
    tv(j) = sum(abs(diff(pm)), 'all');
end
set_param(mdl, 'FastRestart', 'off');
save(fullfile(here, sprintf('v13sl_%s%s_%d.mat', set, strrep(['_' design], '_pole', ''), part)), 'idx', 'peak', 'peakx', 'ts', 'tsx', 'tv');
fprintf('v13 %s part %d: %d sims in %.0f s\n', set, part, n, toc(t0));
end
