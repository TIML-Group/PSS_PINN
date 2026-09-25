function L = case30_lib()
% case30_lib  Shared functions for the 30-bus workflow.
%   L = case30_lib();   then e.g.  Pk = L.scenario_P(...)
% Every script uses these, so scenarios are built the same way everywhere.
L.kron_reduce   = @kron_reduce;
L.open_branch   = @open_branch;
L.scenario_P    = @scenario_P;
L.pack_states   = @pack_states;
L.state_names   = @state_names;
L.noise_vector  = @noise_vector;
L.flat          = @flat;
L.fmt_time      = @fmt_time;
L.solve_delta_p = @solve_delta_p;
L.fast_traj     = @fast_traj;
end

%% ======================= network =======================
function [Yr, Vr] = kron_reduce(Y, gb, Xdp)
% Reduce the network to the machine internal nodes (behind jXd').
NB  = size(Y, 1);  N = numel(gb);
yg  = 1 ./ (1j * Xdp(:));
Ybb = Y;
Ybg = zeros(NB, N);
for i = 1:N
    Ybb(gb(i), gb(i)) = Ybb(gb(i), gb(i)) + yg(i);
    Ybg(gb(i), i)     = -yg(i);
end
Vr = -(Ybb \ Ybg);             % V_bus = Vr * E'
Yr = diag(yg) + Ybg.' * Vr;    % I_gen = Yr * E'
end

function Y = open_branch(Y, B, i)
% Remove branch i exactly (series, charging, tap), using the JSON line data.
f = B.br_from(i);  t = B.br_to(i);
Y(f, f) = Y(f, f) - B.br_Yff(i);
Y(f, t) = Y(f, t) - B.br_Yft(i);
Y(t, f) = Y(t, f) - B.br_Ytf(i);
Y(t, t) = Y(t, t) - B.br_Ytt(i);
end

function Pk = scenario_P(P0, B, Yr_pre, Vr_pre, typ, a, lc, tf, tc)
% Parameter struct P for one scenario.
%   gen : machine lc, set-point changes by -a * Pm0(lc) during the fault
%   load: Y(lc,lc) += a during the fault
%   line: branch lc open during the fault (a = 1), reconnected at clearing
NG = numel(P0.H);
Pk = P0;
Pk.Yred_pre  = Yr_pre;  Pk.Vrec_pre  = Vr_pre;
Pk.Yred_post = Yr_pre;  Pk.Vrec_post = Vr_pre;
Pk.dPm0 = zeros(NG, 1);
Yf = B.Y_pre;
switch typ
    case 'gen',  Pk.dPm0(lc) = -a * P0.Pm0(lc);
    case 'load', Yf(lc, lc) = Yf(lc, lc) + a;
    case 'line', if a == 1, Yf = open_branch(B.Y_pre, B, lc); end
    otherwise,   error('Unknown fault type "%s"', typ);
end
[Pk.Yred_fault, Pk.Vrec_fault] = kron_reduce(Yf, B.gen_bus(:), P0.Xdp);
Pk.t_fault = tf;
Pk.t_clear = tc;
end

%% ======================= data layout =======================
function S = pack_states(o, tg, NG, NB)
% Logged signals -> [T x (5*NG + 2*NB)], Python DAE layout:
% delta/omega interleaved, Pm, Id/Iq interleaved, Vm/theta interleaved.
g = @(name) interp1(o.(name).Time, flat(o.(name)), tg);
S = zeros(numel(tg), 5 * NG + 2 * NB);
S(:, 1:2:2 * NG)                 = g('delta');
S(:, 2:2:2 * NG)                 = g('omega');
S(:, 2 * NG + (1:NG))            = g('Pm');
S(:, 3 * NG + 1:2:5 * NG)        = g('Id');
S(:, 3 * NG + 2:2:5 * NG)        = g('Iq');
S(:, 5 * NG + 1:2:end)           = g('Vm');
S(:, 5 * NG + 2:2:end)           = unwrap(g('Th'));
end

function s = state_names(NG, NB)
c = {};
for i = 1:NG, c = [c, {sprintf('delta%d', i), sprintf('omega%d', i)}]; end %#ok<AGROW>
for i = 1:NG, c = [c, {sprintf('Pm%d', i)}]; end                           %#ok<AGROW>
for i = 1:NG, c = [c, {sprintf('Id%d', i), sprintf('Iq%d', i)}]; end       %#ok<AGROW>
for b = 1:NB, c = [c, {sprintf('Vm%d', b), sprintf('Th%d', b)}]; end       %#ok<AGROW>
s = strjoin(c, ',');
end

function sig = noise_vector(NS, NG, NB)
sig = zeros(1, 5 * NG + 2 * NB);
sig(1:2:2 * NG)          = NS.delta;
sig(2:2:2 * NG)          = NS.omega;
sig(2 * NG + (1:NG))     = NS.Pm;
sig(3 * NG + 1:2:5 * NG) = NS.Id;
sig(3 * NG + 2:2:5 * NG) = NS.Iq;
sig(5 * NG + 1:2:end)    = NS.Vm;
sig(5 * NG + 2:2:end)    = NS.Th;
end

function y = flat(ts)
y = squeeze(ts.Data);
if size(y, 1) ~= numel(ts.Time), y = y.'; end
end

function str = fmt_time(sec)
sec = max(0, round(sec));
str = sprintf('%02d:%02d:%02d', floor(sec / 3600), floor(mod(sec, 3600) / 60), mod(sec, 60));
end

%% ======================= fast model (same equations as Simulink) =======================
function C = fast_consts(Pk)
NG     = numel(Pk.H);
C.NG   = NG;
C.Ec   = (Pk.Ed(:) + 1j * Pk.Eq(:)) * exp(-1j * pi / 2);
C.D    = Pk.D(:);
C.i2H  = 1 ./ (2 * Pk.H(:));
C.wb   = 2 * pi * Pk.f0;
C.invR = Pk.invR(:);
C.Tg   = Pk.Tg;
C.x0   = [Pk.delta0(:); ones(NG, 1); Pk.Pm0(:)];
end

function dx = f_rhs(x, Yr, Ps, C)
n  = C.NG;
E  = C.Ec .* exp(1j * x(1:n, :));
Pe = real(E .* conj(Yr * E));
dw = x(n + 1:2 * n, :) - 1;
pm = x(2 * n + 1:3 * n, :);
dx = [C.wb * dw;
      (pm - Pe - C.D .* dw) .* C.i2H;
      (Ps - C.invR .* dw - pm) / C.Tg];
end

function [Yr, Ps] = seg_params(tm, Pk, ta, tr, dP)
if tm < Pk.t_fault
    Yr = Pk.Yred_pre;
elseif tm < Pk.t_clear
    Yr = Pk.Yred_fault;
else
    Yr = Pk.Yred_post;
end
Ps = Pk.Pm0(:) + Pk.dPm0(:) * (tm >= Pk.t_fault && tm < Pk.t_clear);
if tm >= ta && tm < tr
    Ps = Ps + dP;
end
end

function [x, mw, ma] = rk4_segment(x, t0, t1, Yr, Ps, C, hmax, tchk, mw, ma)
if t1 <= t0 + 1e-12, return; end
n = ceil((t1 - t0) / hmax - 1e-9);
h = (t1 - t0) / n;
w = C.NG + 1:2 * C.NG;
for i = 1:n
    k1 = f_rhs(x, Yr, Ps, C);
    k2 = f_rhs(x + h / 2 * k1, Yr, Ps, C);
    k3 = f_rhs(x + h / 2 * k2, Yr, Ps, C);
    k4 = f_rhs(x + h * k3, Yr, Ps, C);
    x  = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    dev = max(abs(x(w, :) - 1), [], 1);
    ma  = max(ma, dev);
    if t0 + i * h >= tchk - 1e-9
        mw = max(mw, dev);
    end
end
end

function [t, w] = fast_traj(Pk, dP, ta, tr, h, T)
% Trajectory of omega (one scenario) -- used by case30_check_model.m
C  = fast_consts(Pk);
bp = unique([0, Pk.t_fault, Pk.t_clear, ta, tr, T]);
bp = bp(bp <= T);
x  = C.x0;  t = 0;  w = x(C.NG + 1:2 * C.NG).';
for i = 1:numel(bp) - 1
    [Yr, Ps] = seg_params((bp(i) + bp(i + 1)) / 2, Pk, ta, tr, dP);
    n  = max(1, ceil((bp(i + 1) - bp(i)) / h - 1e-9));
    hh = (bp(i + 1) - bp(i)) / n;
    for j = 1:n
        k1 = f_rhs(x, Yr, Ps, C);
        k2 = f_rhs(x + hh / 2 * k1, Yr, Ps, C);
        k3 = f_rhs(x + hh / 2 * k2, Yr, Ps, C);
        k4 = f_rhs(x + hh * k3, Yr, Ps, C);
        x  = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        t(end + 1, 1) = bp(i) + j * hh;                       %#ok<AGROW>
        w(end + 1, :) = x(C.NG + 1:2 * C.NG).';               %#ok<AGROW>
    end
end
end

%% ======================= delta_p search =======================
function [status, dp, info] = solve_delta_p(Pk, opt)
% Minimum-norm delta_p (only on opt.mask machines) with
% max|omega - 1| <= limit - margin from t_apply + grace to the end.
% opt fields: ta, tr, T_END, h, grace, limit, margin, mask, dp_max, n_dirs,
%             n_scales, refine_angles, refine_dirs, refine_scales, seed
NG = numel(Pk.H);
C  = fast_consts(Pk);
S  = prepare(Pk, C, opt);
[bw, ba] = evaluate(S, Pk, C, zeros(NG, 1), opt);
info = struct('base_all', ba, 'base_win', bw, 'best_win', bw);
dp   = zeros(NG, 1);
tgt  = opt.limit - opt.margin;
if bw <= tgt
    status = 2;                        % nothing to correct once delta_p can act
    return;
end
if opt.tr <= opt.ta + 1e-9
    status = 3;                        % cleared before delta_p could act
    return;
end
rs    = RandStream('mt19937ar', 'Seed', opt.seed);
mask  = opt.mask(:);
smax  = opt.dp_max * sqrt(nnz(mask));
U     = base_dirs(mask, opt.n_dirs, rs);
s     = linspace(smax / opt.n_scales, smax, opt.n_scales);
found = false;
for pass = 0:numel(opt.refine_angles)
    if pass > 0
        if ~found, break; end
        ds = s(2) - s(1);
        U  = cap_dirs(u_best, deg2rad(opt.refine_angles(pass)), opt.refine_dirs, mask, rs);
        s  = linspace(max(s_best - 2 * ds, ds / opt.refine_scales), s_best, opt.refine_scales);
    end
    cand = reshape(reshape(U, NG, [], 1) .* reshape(s, 1, 1, []), NG, []);
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
    elseif pass == 0                   % best effort
        mw(~inb) = inf;
        [~, j] = min(mw + 1e-9 * nrm);
        info.best_win = mw(j);
        dp = cand(:, j);
    end
end
status = 1 + 2 * ~found;
end

function S = prepare(Pk, C, opt)
S.ta   = opt.ta;
S.tr   = opt.tr;
S.tchk = opt.ta + opt.grace;
NG = C.NG;
bp = unique([0, Pk.t_fault, min(Pk.t_clear, S.ta), S.ta]);
x  = C.x0;  ma = 0;
for i = 1:numel(bp) - 1
    [Yr, Ps] = seg_params((bp(i) + bp(i + 1)) / 2, Pk, S.ta, S.tr, zeros(NG, 1));
    [x, ~, ma] = rk4_segment(x, bp(i), bp(i + 1), Yr, Ps, C, opt.h, inf, 0, ma);
end
S.xa     = x;
S.ma_pre = ma;
c = [S.tchk, Pk.t_clear, S.tr];
S.post_bp = unique([S.ta, opt.T_END, c(c > S.ta & c < opt.T_END)]);
end

function [mw, ma] = evaluate(S, Pk, C, dP, opt)
M  = size(dP, 2);
x  = repmat(S.xa, 1, M);
mw = zeros(1, M);
ma = repmat(S.ma_pre, 1, M);
bp = S.post_bp;
for i = 1:numel(bp) - 1
    [Yr, Ps] = seg_params((bp(i) + bp(i + 1)) / 2, Pk, S.ta, S.tr, dP);
    [x, mw, ma] = rk4_segment(x, bp(i), bp(i + 1), Yr, Ps, C, opt.h, S.tchk, mw, ma);
end
end

function U = base_dirs(mask, K, rs)
% Search directions in the space of the controllable machines
g = find(mask);  n = numel(g);
switch n
    case 1
        V = [1, -1];
    case 2
        a = (0:K - 1) * 2 * pi / K;
        V = [cos(a); sin(a)];
    case 3
        i = (0:K - 1) + 0.5;
        phi = acos(1 - 2 * i / K);  th = pi * (1 + sqrt(5)) * i;
        V = [cos(th) .* sin(phi); sin(th) .* sin(phi); cos(phi)];
    otherwise
        V = randn(rs, n, K);
        V = V ./ vecnorm(V);
end
if n > 1
    V = [V, eye(n), -eye(n), ones(n, 1) / sqrt(n), -ones(n, 1) / sqrt(n)];
end
U = zeros(numel(mask), size(V, 2));
U(g, :) = V;
U = unique(round(U.', 10), 'rows', 'stable').';
end

function U = cap_dirs(u, theta, K, mask, rs)
% u plus K directions within angle theta of u (in the controllable subspace)
g = find(mask);
if numel(g) == 1, U = u; return; end
V = zeros(numel(u), K);
V(g, :) = randn(rs, numel(g), K);
V = V - u * (u.' * V);
nv = vecnorm(V);
keep = nv > 1e-9;
V = V(:, keep) ./ nv(keep);
ph = theta * rand(rs, 1, size(V, 2));
U = [u, u * cos(ph) + V .* sin(ph)];
end
