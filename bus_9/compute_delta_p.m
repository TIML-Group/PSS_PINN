%% compute_delta_p.m
% For every out-of-limit (OOL) scenario in pinn_dataset.mat, find the smallest
% corrective governor set-point change delta_p = [dP1 dP2 dP3] (pu) that keeps
% |omega - 1| <= limit, applied as soon as the fault is detected.
%
% Same idea as delta_pm_solver.py (minimum-norm constant delta_Pm added to the
% governor set-point, hard barrier: any stable solution beats any unstable one),
% with three differences:
%   * the true simulated scenario is used instead of the PINN hand-off state
%   * delta_p starts at detection (t_fault + DETECT_DELAY), not at clearing
%   * the search is a vectorised direction/scale search on the same equations
%     as the Simulink model (thousands of candidates per integration), which
%     finds the minimum-norm solution without gradients; a sample of the
%     results is then re-simulated in Simulink as a check.
%
% Output labels (also appended to pinn_dataset.mat):
%   y_delta_p [N x 3]  corrective delta_p per generator (0 if no action needed)
%   dp_status [N x 1]  0 = not OOL                     -> delta_p = 0
%                      1 = corrected                   -> delta_p found
%                      2 = OOL only before delta_p can act, fine afterwards -> 0
%                      3 = no delta_p within limits fixes it -> best effort

clear; clc;

%% ======================== SETTINGS ========================
DATA_FILE         = 'pinn_dataset.mat';
OUT_FILE          = 'delta_p_labels.mat';
APPEND_TO_DATASET = true;          % also write the labels into DATA_FILE

DETECT_DELAY    = 0.05;            % [s] fault detection -> delta_p applied
HOLD_MODE       = 'until_clear';   % 'until_clear': delta_p withdrawn when the fault clears
                                   % 'to_end'     : delta_p kept until the end of the run
GRACE           = 0.2;             % [s] stability checked from t_apply + GRACE (= Tg,
                                   %     as grace_steps in delta_pm_solver.py)
SAFETY_MARGIN   = 1e-4;            % optimise for limit - margin (e.g. 0.0049)
CORRECTION_GENS = [1 2 3];         % generators allowed to change their set-point
DP_MAX          = 1.0;             % |delta_p| limit per generator [pu]

H_OPT         = 0.01;              % RK4 step for the search [s]
N_DIRS        = 80;                % pass 1: directions on the sphere
N_SCALES      = 30;                % pass 1: magnitudes per direction
REFINE_ANGLES = [20 5];            % refinement cones [deg]
REFINE_DIRS   = 40;
REFINE_SCALES = 25;
CHUNK         = 50;                % progress is printed after each chunk

VERIFY_N       = 500;              % corrected scenarios re-simulated in Simulink
                                   % (0 = skip, inf = all)
VERIFY_RETRIES = 3;                % if a check fails: scale delta_p by VERIFY_SCALE
VERIFY_SCALE   = 1.02;
%% ==========================================================

%% ---------- 1. load dataset labels and system data ----------
fprintf('Loading %s ...\n', DATA_FILE);
mf        = matfile(DATA_FILE);
y_type    = double(mf.y_type(:, 1));
alpha     = mf.meta_alpha;
loc       = mf.meta_loc;
lines     = mf.lines;
t_fault   = mf.y_t_fault;
t_clear   = mf.meta_t_clear;
y_ool     = logical(mf.y_ool);
failed    = logical(mf.failed);
maxdev_ds = mf.meta_maxdev;
tg        = mf.t;
DT        = mf.dt_solver;
LIMIT     = mf.omega_limit;
names     = strsplit(mf.type_names, ',');
T_END     = tg(end);
N         = numel(y_type);

P0 = load('ieee9_gov_simulink_data.mat');
P0.dPc = zeros(3, 1);  P0.t_corr = 1e9;  P0.t_corr_end = 1e9;
B  = load('ieee9_base.mat');
[Yr_pre, Vr_pre] = kron_reduce(B.Y_pre, B.gen_bus(:), P0.Xdp);

t_apply = round((t_fault + DETECT_DELAY) / DT) * DT;   % on the solver grid
if strcmp(HOLD_MODE, 'until_clear')
    t_remove = t_clear;
else
    t_remove = 1e9 * ones(N, 1);
end

opt = struct('T_END', T_END, 'h', H_OPT, 'grace', GRACE, 'limit', LIMIT, ...
             'margin', SAFETY_MARGIN, 'gens', CORRECTION_GENS, 'dp_max', DP_MAX, ...
             'n_dirs', N_DIRS, 'n_scales', N_SCALES, 'refine_angles', REFINE_ANGLES, ...
             'refine_dirs', REFINE_DIRS, 'refine_scales', REFINE_SCALES, ...
             'hold_to_end', strcmp(HOLD_MODE, 'to_end'), 'ta', NaN);

%% ---------- 2. search delta_p for every OOL scenario ----------
delta_p   = zeros(N, 3);
dp_status = zeros(N, 1);
base_all  = nan(N, 1);     % max |omega-1| over the whole run, no correction (fast model)
base_win  = nan(N, 1);     % same, inside the check window
best_win  = nan(N, 1);     % with the chosen delta_p, inside the check window

todo = find(y_ool & ~failed);
fprintf('%d of %d scenarios are OOL -> searching delta_p (applied %g s after the fault, %s)\n', ...
        numel(todo), N, DETECT_DELAY, HOLD_MODE);
fprintf('%-18s %7s %10s %10s %8s %8s %8s\n', 'scenarios', 'done', 'elapsed', 'remaining', ...
        'fixed', 'no-act', 'failed');
t_start = tic;
for c0 = 1:CHUNK:numel(todo)
    ids = todo(c0:min(c0 + CHUNK - 1, numel(todo)));
    n   = numel(ids);
    r_dp = zeros(n, 3);  r_st = zeros(n, 1);
    r_ba = zeros(n, 1);  r_bw = zeros(n, 1);  r_best = zeros(n, 1);
    parfor j = 1:n       % runs serially without the Parallel Computing Toolbox
        k  = ids(j);
        Pk = scenario_P(P0, B, Yr_pre, Vr_pre, names{y_type(k)}, alpha(k), loc(k), ...
                        lines, t_fault(k), t_clear(k));
        o  = opt;
        o.ta = t_apply(k);
        [st, d, info] = solve_delta_p(Pk, o);
        r_dp(j, :) = d.';
        r_st(j)    = st;
        r_ba(j)    = info.base_all;
        r_bw(j)    = info.base_win;
        r_best(j)  = info.best_win;
    end
    delta_p(ids, :) = r_dp;   dp_status(ids) = r_st;
    base_all(ids)   = r_ba;   base_win(ids)  = r_bw;   best_win(ids) = r_best;

    done = c0 + n - 1;
    el   = toc(t_start);
    st_done = dp_status(todo(1:done));
    fprintf('%6d / %-9d %6.1f%% %10s %10s %8d %8d %8d\n', done, numel(todo), ...
            100 * done / numel(todo), fmt_time(el), fmt_time(el / done * (numel(todo) - done)), ...
            nnz(st_done == 1), nnz(st_done == 2), nnz(st_done == 3));
end

% consistency: fast model vs the Simulink runs that produced the dataset
if ~isempty(todo)
    dd = abs(base_all(todo) - maxdev_ds(todo));
    fprintf('\nFast model vs dataset (no correction): max |diff| of peak deviation = %.2e pu\n', max(dd));
end

%% ---------- 3. check a sample in Simulink ----------
verify_maxdev = nan(N, 1);
fixed = find(dp_status == 1);
if VERIFY_N > 0 && ~isempty(fixed)
    rng(0);
    pending = fixed(randperm(numel(fixed), min(VERIFY_N, numel(fixed))));
    fprintf('\nVerifying %d corrected scenarios in Simulink ...\n', numel(pending));
    mdl = 'ieee9_governor';
    if bdIsLoaded(mdl), close_system(mdl, 0); end
    load_system(mdl);
    set_param(mdl, 'StopTime', num2str(T_END), 'FixedStep', num2str(DT));
    tw = find_system(mdl, 'BlockType', 'ToWorkspace');
    for i = 1:numel(tw), set_param(tw{i}, 'Decimation', '1'); end
    save_system(mdl);
    P = P0; %#ok<NASGU>

    for attempt = 0:VERIFY_RETRIES
        if isempty(pending), break; end
        for c0 = 1:CHUNK:numel(pending)
            ids = pending(c0:min(c0 + CHUNK - 1, numel(pending)));
            in  = repmat(Simulink.SimulationInput(mdl), numel(ids), 1);
            for j = 1:numel(ids)
                k  = ids(j);
                Pk = scenario_P(P0, B, Yr_pre, Vr_pre, names{y_type(k)}, alpha(k), loc(k), ...
                                lines, t_fault(k), t_clear(k));
                Pk.dPc = delta_p(k, :).';
                Pk.t_corr = t_apply(k);
                Pk.t_corr_end = t_remove(k);
                in(j) = in(j).setVariable('P', Pk);
            end
            outs = parsim(in, 'UseFastRestart', 'on', 'ShowProgress', 'off', ...
                          'StopOnError', 'off', 'TransferBaseWorkspaceVariables', 'off');
            for j = 1:numel(ids)
                k = ids(j);
                if ~isempty(outs(j).ErrorMessage)
                    warning('Scenario %d: %s', k, outs(j).ErrorMessage);
                    continue;
                end
                w  = flat(outs(j).omega);
                tt = outs(j).omega.Time;
                m  = tt >= t_apply(k) + GRACE - 1e-9;
                verify_maxdev(k) = max(abs(w(m, :) - 1), [], 'all');
            end
            clear in outs
        end
        bad = pending(~(verify_maxdev(pending) <= LIMIT));
        fprintf('  attempt %d: %d / %d within the limit\n', attempt + 1, ...
                numel(pending) - numel(bad), numel(pending));
        if attempt < VERIFY_RETRIES
            delta_p(bad, :) = max(-DP_MAX, min(DP_MAX, delta_p(bad, :) * VERIFY_SCALE));
        else
            dp_status(bad) = 3;
        end
        pending = bad;
    end
    set_param(mdl, 'FastRestart', 'off');
    v = ~isnan(verify_maxdev);
    fprintf('Simulink check: max deviation %.5f (limit %.3f), %d of %d pass\n', ...
            max(verify_maxdev(v)), LIMIT, nnz(verify_maxdev(v) <= LIMIT), nnz(v));
end

%% ---------- 4. summary ----------
fprintf('\n%-6s %8s %8s %8s %8s %8s %12s\n', 'type', 'total', 'not OOL', 'fixed', ...
        'no-act', 'failed', 'mean |dP|');
for c = 1:numel(names)
    m = y_type == c & ~failed;
    f = m & dp_status == 1;
    fprintf('%-6s %8d %8d %8d %8d %8d %12.3f\n', names{c}, nnz(m), nnz(m & dp_status == 0), ...
            nnz(f), nnz(m & dp_status == 2), nnz(m & dp_status == 3), ...
            mean(sqrt(sum(delta_p(f, :).^2, 2))));
end

%% ---------- 5. save ----------
L.y_delta_p          = delta_p;
L.dp_status          = int8(dp_status);
L.t_apply            = t_apply;
L.t_remove           = t_remove;
L.base_maxdev_all    = base_all;
L.base_maxdev_window = base_win;
L.best_maxdev_window = best_win;
L.verify_maxdev      = verify_maxdev;
L.dp_settings        = sprintf(['detect_delay=%g, hold=%s, grace=%g, margin=%g, ' ...
                                'gens=%s, dp_max=%g'], DETECT_DELAY, HOLD_MODE, GRACE, ...
                                SAFETY_MARGIN, mat2str(CORRECTION_GENS), DP_MAX);
save(OUT_FILE, '-struct', 'L');
fprintf('\nSaved %s\n', OUT_FILE);
if APPEND_TO_DATASET
    mw = matfile(DATA_FILE, 'Writable', true);
    fn = fieldnames(L);
    for i = 1:numel(fn), mw.(fn{i}) = L.(fn{i}); end
    fprintf('Added delta_p labels to %s\n', DATA_FILE);
end


%% ======================== helpers ========================
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
