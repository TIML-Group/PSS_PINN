function gen_step_labels(worker, n_workers, varargin)
%GEN_STEP_LABELS  Per-time-step "first action now" dP labels for the 9-bus set.
%
% For every recorded step t_i (dt_solver grid) inside [t_fault, t_clear) of
% each scenario in scenarios.mat, run the EXISTING compute_delta_p.m optimizer
% (solve_delta_p, verbatim) with t_apply = t_i: start from the uncorrected
% trajectory at t_i, dP held from t_i until clearing, smallest-norm dP with
% max|omega-1| <= limit - margin from t_i + grace to the end of the run.
% Every step is an independent trial (prepare() re-integrates the uncorrected
% run up to t_i); nothing carries over between steps.
%
% Steps before the fault / after clearing are dP = 0 by definition and are
% not stored. Per stored step:
%   status 0 = scenario never out of limit (y_ool false, as compute_delta_p.m),
%   status 1 = corrected, 2 = no action needed (dP = 0),
%          3 = no feasible dP within |dP| <= 1 (best-effort dP kept, FLAG)
%
% Batches of BATCH scenarios; batch b goes to worker mod(b-1, n_workers).
% A batch file that exists is skipped, so a killed job resumes where it left.
%
%   gen_step_labels(0, 1, 'mode', 'check50', 'n', 200)   % 50 ms reproduction
%   gen_step_labels(w, W)                                % full job, worker w
p = inputParser;
addParameter(p, 'batch', 10);
addParameter(p, 'mode', 'full');
addParameter(p, 'n', 200);
addParameter(p, 'max_batches', inf);
parse(p, varargin{:});
BATCH = p.Results.batch;

here = fileparts(mfilename('fullpath'));
bus9 = fullfile(here, '..', '..');
S0   = load(fullfile(here, 'scenarios.mat'));        % ks (1-based), from Python split
ks_all = double(S0.ks(:));

% ---- settings: identical to compute_delta_p.m ----
DETECT_DELAY = 0.05; GRACE = 0.2; SAFETY_MARGIN = 1e-4;
CORRECTION_GENS = [1 2 3]; DP_MAX = 1.0; H_OPT = 0.01;
N_DIRS = 80; N_SCALES = 30; REFINE_ANGLES = [20 5]; REFINE_DIRS = 40; REFINE_SCALES = 25;

mf      = matfile(fullfile(bus9, 'pinn_dataset.mat'));
y_type  = double(mf.y_type(:, 1));
alpha   = mf.meta_alpha;  loc = mf.meta_loc;  lines = mf.lines;
t_fault = mf.y_t_fault;   t_clear = mf.meta_t_clear;  y_ool = logical(mf.y_ool);
tg      = mf.t;  DT = mf.dt_solver;  LIMIT = mf.omega_limit;
names   = strsplit(mf.type_names, ',');
T_END   = tg(end);

P0 = load(fullfile(bus9, 'ieee9_gov_simulink_data.mat'));
P0.dPc = zeros(3, 1);  P0.t_corr = 1e9;  P0.t_corr_end = 1e9;
B  = load(fullfile(bus9, 'ieee9_base.mat'));
[Yr_pre, Vr_pre] = kron_reduce(B.Y_pre, B.gen_bus(:), P0.Xdp);
opt = struct('T_END', T_END, 'h', H_OPT, 'grace', GRACE, 'limit', LIMIT, ...
             'margin', SAFETY_MARGIN, 'gens', CORRECTION_GENS, 'dp_max', DP_MAX, ...
             'n_dirs', N_DIRS, 'n_scales', N_SCALES, 'refine_angles', REFINE_ANGLES, ...
             'refine_dirs', REFINE_DIRS, 'refine_scales', REFINE_SCALES, ...
             'hold_to_end', false, 'ta', NaN);

if strcmp(p.Results.mode, 'check50')
    % ---- reproduce the original labels at t_apply = round((tf+0.05)/DT)*DT ----
    rng(1); ids = ks_all(randperm(numel(ks_all), min(p.Results.n, numel(ks_all))));
    old = load(fullfile(bus9, 'delta_p_labels.mat'), 'y_delta_p', 'dp_status', 't_apply', 'verify_maxdev');
    dp = zeros(numel(ids), 3); st = zeros(numel(ids), 1); tt = tic;
    for j = 1:numel(ids)
        k = ids(j);
        Pk = scenario_P(P0, B, Yr_pre, Vr_pre, names{y_type(k)}, alpha(k), loc(k), lines, t_fault(k), t_clear(k));
        o = opt; o.ta = round((t_fault(k) + DETECT_DELAY) / DT) * DT;
        [s, d] = solve_delta_p(Pk, o);
        dp(j, :) = d.'; st(j) = s;
    end
    el = toc(tt);
    R.ids = ids; R.dp = dp; R.status = st; R.old_dp = old.y_delta_p(ids, :);
    R.old_status = double(old.dp_status(ids)); R.old_verified = ~isnan(old.verify_maxdev(ids));
    R.seconds = el;
    save(fullfile(here, 'check50.mat'), '-struct', 'R');
    fprintf('check50: %d scenarios, %.1f s\n', numel(ids), el);
    return;
end

outdir = fullfile(here, 'batches');  if ~exist(outdir, 'dir'), mkdir(outdir); end
nb = ceil(numel(ks_all) / BATCH);  done_here = 0;
for b = 1:nb
    if mod(b - 1, n_workers) ~= worker, continue; end
    f = fullfile(outdir, sprintf('batch_%05d.mat', b));
    if exist(f, 'file'), continue; end
    if done_here >= p.Results.max_batches, break; end
    ids = ks_all((b - 1) * BATCH + 1 : min(b * BATCH, numel(ks_all)));
    K = []; I = []; DP = []; ST = []; BW = []; BS = []; tt = tic;
    for k = ids.'
        Pk = scenario_P(P0, B, Yr_pre, Vr_pre, names{y_type(k)}, alpha(k), loc(k), lines, t_fault(k), t_clear(k));
        steps = find(tg >= t_fault(k) - 1e-9 & tg < t_clear(k) - 1e-9);   % 1-based grid index
        n = numel(steps);
        dpk = zeros(n, 3); stk = zeros(n, 1); bwk = nan(n, 1); bsk = nan(n, 1);
        % same gate as compute_delta_p.m: a scenario that never leaves the limit
        % (y_ool false) gets dP = 0, status 0, at every step -- no search
        n_search = n * y_ool(k);
        for j = 1:n_search
            o = opt; o.ta = tg(steps(j));
            [s, d, info] = solve_delta_p(Pk, o);
            dpk(j, :) = d.'; stk(j) = s; bwk(j) = info.base_win; bsk(j) = info.best_win;
        end
        K = [K; k * ones(n, 1)]; I = [I; steps(:)]; DP = [DP; dpk];  %#ok<AGROW>
        ST = [ST; stk]; BW = [BW; bwk]; BS = [BS; bsk];             %#ok<AGROW>
    end
    L.k = int32(K); L.step = int32(I); L.dp = single(DP); L.status = int8(ST);
    L.base_win = single(BW); L.best_win = single(BS); L.seconds = toc(tt);
    L.scenarios = int32(ids);
    tmp = [f '.tmp.mat'];  save(tmp, '-struct', 'L');  movefile(tmp, f);   % atomic: no half files
    done_here = done_here + 1;
    fprintf('worker %d batch %d/%d: %d steps, %.1f s\n', worker, b, nb, numel(K), L.seconds);
end
end

function Pk = scenario_P(P0, B, Yr_pre, Vr_pre, typ, a, lc, lines, tf, tc)
% Same network/set-point changes as generate_dataset.m
Pk = P0;
Pk.Yred_pre  = Yr_pre;  Pk.Vrec_pre  = Vr_pre;
Pk.Yred_post = Yr_pre;  Pk.Vrec_post = Vr_pre;
Pk.dPm0 = zeros(3, 1);
Yf = B.Y_pre;
switch typ
    case 'gen'
        Pk.dPm0(lc) = -a * P0.Pm0(lc);
    case 'load'
        Yf(lc, lc) = Yf(lc, lc) + a;
    case 'line'
        if a == 1, Yf = open_line_Y(B.Y_pre, lines(lc, 1), lines(lc, 2)); end
    otherwise
        error('Unknown fault type "%s"', typ);
end
[Pk.Yred_fault, Pk.Vrec_fault] = kron_reduce(Yf, B.gen_bus(:), P0.Xdp);
Pk.t_fault = tf;
Pk.t_clear = tc;
end

function [status, dp, info] = solve_delta_p(Pk, opt)
% Minimum-norm delta_p with max|omega-1| <= limit - margin in the check window.
C = fast_consts(Pk);
S = prepare(Pk, C, opt);
[bw, ba] = evaluate(S, Pk, C, zeros(3, 1), opt);
info.base_all = ba;
info.base_win = bw;
info.best_win = bw;
dp  = zeros(3, 1);
tgt = opt.limit - opt.margin;
if bw <= tgt
    status = 2;                      % nothing to correct once delta_p can act
    return;
end
if ~opt.hold_to_end && opt.ta >= Pk.t_clear - 1e-9
    status = 3;                      % fault already cleared before delta_p could act
    return;
end

mask = zeros(3, 1);  mask(opt.gens) = 1;
U = clean_dirs([fib_sphere(opt.n_dirs), eye(3), -eye(3), ...
                ones(3, 1) / sqrt(3), -ones(3, 1) / sqrt(3)], mask);
smax  = opt.dp_max * sqrt(nnz(mask));
s     = linspace(smax / opt.n_scales, smax, opt.n_scales);
found = false;
for pass = 0:numel(opt.refine_angles)
    if pass > 0
        if ~found, break; end
        ds = s(2) - s(1);
        U  = clean_dirs(cap_dirs(u_best, deg2rad(opt.refine_angles(pass)), opt.refine_dirs), mask);
        s  = linspace(max(s_best - 2 * ds, ds / opt.refine_scales), s_best, opt.refine_scales);
    end
    cand = reshape(U, 3, [], 1) .* reshape(s, 1, 1, []);    % 3 x K x J
    cand = reshape(cand, 3, []);
    inb  = all(abs(cand) <= opt.dp_max + 1e-12, 1);
    mw   = evaluate(S, Pk, C, cand, opt);
    nrm  = sqrt(sum(cand .^ 2, 1));
    feas = (mw <= tgt) & inb;
    if any(feas)
        nrm(~feas) = inf;
        [s_best, j] = min(nrm);
        dp     = cand(:, j);
        u_best = dp / s_best;
        found  = true;
        info.best_win = mw(j);
    elseif pass == 0                 % best effort: smallest remaining deviation
        mw(~inb) = inf;
        [~, j] = min(mw + 1e-9 * nrm);   % ties -> smaller delta_p
        info.best_win = mw(j);
        dp = cand(:, j);
    end
end
status = 1 + 2 * ~found;
end

function C = fast_consts(Pk)
C.Ec  = (Pk.Ed(:) + 1j * Pk.Eq(:)) * exp(-1j * pi / 2);
C.D   = Pk.D(:);
C.i2H = 1 ./ (2 * Pk.H(:));
C.wb  = 2 * pi * Pk.f0;
C.R   = Pk.R;
C.Tg  = Pk.Tg;
C.x0  = [Pk.delta0(:); ones(3, 1); Pk.Pm0(:)];
end

function S = prepare(Pk, C, opt)
% Integrate once (no correction) up to t_apply; candidates start from there.
S.ta   = opt.ta;
S.tchk = opt.ta + opt.grace;
S.hold_to_end = opt.hold_to_end;
bp = unique([0, Pk.t_fault, min(Pk.t_clear, S.ta), S.ta]);
x = C.x0;  ma = 0;
for i = 1:numel(bp) - 1
    [Yr, Ps] = seg_params((bp(i) + bp(i + 1)) / 2, Pk, S, zeros(3, 1));
    [x, ~, ma] = rk4_segment(x, bp(i), bp(i + 1), Yr, Ps, C, opt.h, inf, 0, ma);
end
S.xa     = x;
S.ma_pre = ma;
c = [S.tchk, Pk.t_clear];
S.post_bp = unique([S.ta, opt.T_END, c(c > S.ta & c < opt.T_END)]);
end

function [mw, ma] = evaluate(S, Pk, C, dP, opt)
M  = size(dP, 2);
x  = repmat(S.xa, 1, M);
mw = zeros(1, M);
ma = repmat(S.ma_pre, 1, M);
bp = S.post_bp;
for i = 1:numel(bp) - 1
    [Yr, Ps] = seg_params((bp(i) + bp(i + 1)) / 2, Pk, S, dP);
    [x, mw, ma] = rk4_segment(x, bp(i), bp(i + 1), Yr, Ps, C, opt.h, S.tchk, mw, ma);
end
end

function [Yr, Ps] = seg_params(tm, Pk, S, dP)
if tm < Pk.t_fault
    Yr = Pk.Yred_pre;
elseif tm < Pk.t_clear
    Yr = Pk.Yred_fault;
else
    Yr = Pk.Yred_post;
end
in_fault = tm >= Pk.t_fault && tm < Pk.t_clear;
Ps = Pk.Pm0(:) + Pk.dPm0(:) * in_fault;
if tm >= S.ta && (S.hold_to_end || tm < Pk.t_clear)
    Ps = Ps + dP;                    % 3 x M
end
end

function [x, mw, ma] = rk4_segment(x, t0, t1, Yr, Ps, C, hmax, tchk, mw, ma)
if t1 <= t0 + 1e-12, return; end
n = ceil((t1 - t0) / hmax - 1e-9);
h = (t1 - t0) / n;
for i = 1:n
    k1 = f_rhs(x, Yr, Ps, C);
    k2 = f_rhs(x + h / 2 * k1, Yr, Ps, C);
    k3 = f_rhs(x + h / 2 * k2, Yr, Ps, C);
    k4 = f_rhs(x + h * k3, Yr, Ps, C);
    x  = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    dev = max(abs(x(4:6, :) - 1), [], 1);
    ma  = max(ma, dev);
    if t0 + i * h >= tchk - 1e-9
        mw = max(mw, dev);
    end
end
end

function dx = f_rhs(x, Yr, Ps, C)
% Same equations as the Simulink model (Kron-reduced network), for 3 x M states
E  = C.Ec .* exp(1j * x(1:3, :));
Pe = real(E .* conj(Yr * E));
dw = x(4:6, :) - 1;
dx = [C.wb * dw;
      (x(7:9, :) - Pe - C.D .* dw) .* C.i2H;
      (Ps - dw / C.R - x(7:9, :)) / C.Tg];
end

function U = fib_sphere(K)
i   = (0:K - 1) + 0.5;
phi = acos(1 - 2 * i / K);
th  = pi * (1 + sqrt(5)) * i;
U   = [cos(th) .* sin(phi); sin(th) .* sin(phi); cos(phi)];
end

function U = cap_dirs(u, theta, K)
% K directions within angle theta of u (u itself first)
i  = (0:K - 1) + 0.5;
z  = 1 - (1 - cos(theta)) * i / K;
ph = pi * (1 + sqrt(5)) * i;
r  = sqrt(max(0, 1 - z .^ 2));
if abs(u(1)) < 0.9, a = cross(u, [1; 0; 0]); else, a = cross(u, [0; 1; 0]); end
a = a / norm(a);
b = cross(u, a);
U = [u, a * (r .* cos(ph)) + b * (r .* sin(ph)) + u * z];
end

function U = clean_dirs(U, mask)
U = U .* mask;
n = sqrt(sum(U .^ 2, 1));
U = U(:, n > 1e-9) ./ n(n > 1e-9);
U = unique(round(U.', 10), 'rows', 'stable').';
end

function [Yr, Vr] = kron_reduce(Y, gb, Xdp)
NB  = size(Y, 1);  N = numel(gb);
yg  = 1 ./ (1j * Xdp(:));
Ybb = Y;
Ybg = zeros(NB, N);
for i = 1:N
    Ybb(gb(i), gb(i)) = Ybb(gb(i), gb(i)) + yg(i);
    Ybg(gb(i), i)     = -yg(i);
end
Vr = -(Ybb \ Ybg);
Yr = diag(yg) + Ybg.' * Vr;
end

function Y = open_line_Y(Y, r, s)
y = -Y(r, s);
Y(r, r) = Y(r, r) - y;  Y(s, s) = Y(s, s) - y;
Y(r, s) = 0;            Y(s, r) = 0;
end

function y = flat(ts)
y = squeeze(ts.Data);
if size(y, 1) ~= numel(ts.Time), y = y.'; end
end

function str = fmt_time(sec)
sec = max(0, round(sec));
str = sprintf('%02d:%02d:%02d', floor(sec / 3600), floor(mod(sec, 3600) / 60), mod(sec, 60));
end
