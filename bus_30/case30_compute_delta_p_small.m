%% case30_compute_delta_p.m
% For every out-of-limit scenario in case30_dataset.mat, find the smallest
% corrective governor set-point change delta_p (one value per machine, non-zero
% only on governed machines) that keeps |omega - 1| <= limit. delta_p is applied
% at fault detection (t_fault + DETECT_DELAY) -- same method as the 9-bus
% compute_delta_p.m.
%
% Labels (also appended to case30_dataset.mat):
%   y_delta_p [N x NG]  corrective delta_p per machine (0 if no action needed)
%   dp_status [N x 1]   0 = not OOL, 1 = corrected, 2 = OOL only before delta_p
%                       can act (delta_p = 0), 3 = not correctable (best effort)

clear; clc;
L = case30_lib();

%% ======================== SETTINGS ========================
DATA_FILE         = 'case30_dataset_small.mat';
OUT_FILE          = 'case30_delta_p_labels_small.mat';
APPEND_TO_DATASET = true;

DETECT_DELAY  = 0.05;             % [s] fault detection -> delta_p applied
HOLD_MODE     = 'until_clear';    % or 'to_end'
GRACE         = 0.2;              % [s] (= Tg)
SAFETY_MARGIN = 1e-4;
DP_MAX        = 1.0;              % |delta_p| per machine [pu]
% machines that may receive delta_p: [] = every machine with a governor
CORRECTION_GENS = [];

H_OPT         = 0.01;
N_DIRS        = 72;               % 2 controllable machines -> 5 deg steps on a circle
N_SCALES      = 30;
REFINE_ANGLES = [20 5];
REFINE_DIRS   = 40;
REFINE_SCALES = 25;
CHUNK         = 50;

VERIFY_N       = 500;             % corrected scenarios re-simulated in Simulink
VERIFY_RETRIES = 3;
VERIFY_SCALE   = 1.02;
%% ==========================================================

fprintf('Loading %s ...\n', DATA_FILE);
mf        = matfile(DATA_FILE);
y_type    = double(mf.y_type(:, 1));
alpha     = mf.meta_alpha;
loc       = mf.meta_loc;
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

P0 = load('case30_simulink_data.mat');
B  = load('case30_base.mat');
NG = numel(P0.H);
[Yr_pre, Vr_pre] = L.kron_reduce(B.Y_pre, B.gen_bus(:), P0.Xdp);
mask = B.governor(:) > 0.5;
if ~isempty(CORRECTION_GENS), mask = false(NG, 1); mask(CORRECTION_GENS) = true; end
fprintf('delta_p on machines %s (buses %s)\n', mat2str(find(mask).'), mat2str(B.gen_bus(mask).'));

t_apply = round((t_fault + DETECT_DELAY) / DT) * DT;
if strcmp(HOLD_MODE, 'until_clear'), t_remove = t_clear; else, t_remove = 1e9 * ones(N, 1); end

opt = struct('T_END', T_END, 'h', H_OPT, 'grace', GRACE, 'limit', LIMIT, ...
             'margin', SAFETY_MARGIN, 'mask', mask, 'dp_max', DP_MAX, ...
             'n_dirs', N_DIRS, 'n_scales', N_SCALES, 'refine_angles', REFINE_ANGLES, ...
             'refine_dirs', REFINE_DIRS, 'refine_scales', REFINE_SCALES, ...
             'ta', NaN, 'tr', NaN, 'seed', 0);

%% ---------- search ----------
delta_p   = zeros(N, NG);
dp_status = zeros(N, 1);
base_all  = nan(N, 1);
base_win  = nan(N, 1);
best_win  = nan(N, 1);

todo = find(y_ool & ~failed);
fprintf('%d of %d scenarios are OOL -> searching delta_p (applied %g s after the fault, %s)\n', ...
        numel(todo), N, DETECT_DELAY, HOLD_MODE);
fprintf('%-18s %7s %10s %10s %8s %8s %8s\n', 'scenarios', 'done', 'elapsed', 'remaining', ...
        'fixed', 'no-act', 'failed');
t_start = tic;
for c0 = 1:CHUNK:numel(todo)
    ids = todo(c0:min(c0 + CHUNK - 1, numel(todo)));
    n   = numel(ids);
    r_dp = zeros(n, NG);  r_st = zeros(n, 1);
    r_ba = zeros(n, 1);   r_bw = zeros(n, 1);  r_best = zeros(n, 1);
    parfor j = 1:n
        k  = ids(j);
        Pk = L.scenario_P(P0, B, Yr_pre, Vr_pre, names{y_type(k)}, alpha(k), loc(k), ...
                          t_fault(k), t_clear(k));
        o  = opt;
        o.ta = t_apply(k);  o.tr = t_remove(k);  o.seed = k;
        [st, dvec, info] = L.solve_delta_p(Pk, o);
        r_dp(j, :) = dvec.';
        r_st(j)    = st;
        r_ba(j)    = info.base_all;
        r_bw(j)    = info.base_win;
        r_best(j)  = info.best_win;
    end
    delta_p(ids, :) = r_dp;   dp_status(ids) = r_st;
    base_all(ids)   = r_ba;   base_win(ids)  = r_bw;   best_win(ids) = r_best;

    done = c0 + n - 1;
    el   = toc(t_start);
    sd   = dp_status(todo(1:done));
    fprintf('%6d / %-9d %6.1f%% %10s %10s %8d %8d %8d\n', done, numel(todo), ...
            100 * done / numel(todo), L.fmt_time(el), L.fmt_time(el / done * (numel(todo) - done)), ...
            nnz(sd == 1), nnz(sd == 2), nnz(sd == 3));
end
if ~isempty(todo)
    dd = abs(base_all(todo) - maxdev_ds(todo));
    fprintf('\nFast model vs dataset (no correction): max |diff| of peak deviation = %.2e pu\n', max(dd));
end

%% ---------- Simulink check ----------
verify_maxdev = nan(N, 1);
fixed = find(dp_status == 1);
if VERIFY_N > 0 && ~isempty(fixed)
    rng(0);
    pending = fixed(randperm(numel(fixed), min(VERIFY_N, numel(fixed))));
    fprintf('\nVerifying %d corrected scenarios in Simulink ...\n', numel(pending));
    mdl = 'case30_governor';
    if bdIsLoaded(mdl), close_system(mdl, 0); end
    load_system(mdl);
    set_param(mdl, 'StopTime', num2str(T_END), 'FixedStep', num2str(DT));
    tw = find_system(mdl, 'BlockType', 'ToWorkspace');
    for i = 1:numel(tw)
        set_param(tw{i}, 'Decimation', '1');
        set_param(tw{i}, 'Commented', 'off');
    end
    save_system(mdl);
    P = P0; %#ok<NASGU>

    for attempt = 0:VERIFY_RETRIES
        if isempty(pending), break; end
        for c0 = 1:CHUNK:numel(pending)
            ids = pending(c0:min(c0 + CHUNK - 1, numel(pending)));
            in  = repmat(Simulink.SimulationInput(mdl), numel(ids), 1);
            for j = 1:numel(ids)
                k  = ids(j);
                Pk = L.scenario_P(P0, B, Yr_pre, Vr_pre, names{y_type(k)}, alpha(k), loc(k), ...
                                  t_fault(k), t_clear(k));
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
                w  = L.flat(outs(j).omega);
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

%% ---------- summary ----------
fprintf('\n%-6s %8s %8s %8s %8s %8s %12s\n', 'type', 'total', 'not OOL', 'fixed', ...
        'no-act', 'failed', 'mean |dP|');
for c = 1:numel(names)
    m = y_type == c & ~failed;
    f = m & dp_status == 1;
    fprintf('%-6s %8d %8d %8d %8d %8d %12.3f\n', names{c}, nnz(m), nnz(m & dp_status == 0), ...
            nnz(f), nnz(m & dp_status == 2), nnz(m & dp_status == 3), ...
            mean(sqrt(sum(delta_p(f, :) .^ 2, 2))));
end

%% ---------- save ----------
Lb.y_delta_p          = delta_p;
Lb.dp_status          = int8(dp_status);
Lb.t_apply            = t_apply;
Lb.t_remove           = t_remove;
Lb.dp_machines        = find(mask);
Lb.base_maxdev_all    = base_all;
Lb.base_maxdev_window = base_win;
Lb.best_maxdev_window = best_win;
Lb.verify_maxdev      = verify_maxdev;
Lb.dp_settings        = sprintf('detect_delay=%g, hold=%s, grace=%g, margin=%g, machines=%s, dp_max=%g', ...
                                DETECT_DELAY, HOLD_MODE, GRACE, SAFETY_MARGIN, ...
                                mat2str(find(mask).'), DP_MAX);
save(OUT_FILE, '-struct', 'Lb');
fprintf('\nSaved %s\n', OUT_FILE);
if APPEND_TO_DATASET
    mw = matfile(DATA_FILE, 'Writable', true);
    fn = fieldnames(Lb);
    for i = 1:numel(fn), mw.(fn{i}) = Lb.(fn{i}); end
    fprintf('Added delta_p labels to %s\n', DATA_FILE);
end
