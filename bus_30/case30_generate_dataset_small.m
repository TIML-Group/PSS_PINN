%% case30_generate_dataset_small.m
% 10,000 randomised fault scenarios on the IEEE 30-bus model.
%
%   Input  X : [N_SCEN x 8001 x 90] noisy full simulation, EVERY 1 ms sample
%   Labels   : y_type, y_t_fault, y_ool
%
% Needs case30_simulink_data.mat, case30_base.mat (export_case30.py) and
% case30_governor.slx (case30_build_model.m).
%
% SIZE: 10000 x 8001 x 90 single = ~28.8 GB. It is written to disk batch by
% batch; put the folder on a local drive with enough space (not OneDrive).

clear; clc;
L = case30_lib();

%% ======================== SETTINGS ========================
N_SCEN      = 100;
T_END       = 8.0;          % [s]
DT          = 1e-3;         % solver step [s]
SAVE_EVERY  = 1;            % 1 = keep every sample (8001 per scenario)
BATCH       = 50;           % scenarios per parsim call
SEED        = 42;
OMEGA_LIMIT = 0.005;        % OOL if |omega - 1| > OMEGA_LIMIT (any machine, any time)
OUT_FILE    = 'case30_dataset_small.mat';
SAVE_CLEAN  = false;        % also save noise-free X_clean (doubles the size)
SHOW_BAR    = true;

% Same fault definitions and ranges as the 9-bus dataset
%   gen : producing machine, set-point changes by -alpha * Pm0 during the fault
%   load: Y(bus,bus) += alpha at a load bus during the fault
%   line: line open during the fault (alpha = 1), reconnected at clearing
FAULT_TYPES      = {'gen', 'load', 'line'};
RANGE.alpha.gen  = [0.10 1.00];
RANGE.alpha.load = [0.10 1.00];
RANGE.alpha.line = [1 1];
RANGE.t_fault    = [0.50 4.00];
RANGE.duration   = [0.05 3.00];

NS.delta = 0.01;  NS.omega = 5e-5; NS.Pm = 0.005;
NS.Id    = 0.005; NS.Iq    = 0.005; NS.Vm = 0.001; NS.Th = 0.01;
%% ==========================================================

P0  = load('case30_simulink_data.mat');
B   = load('case30_base.mat');
mdl = 'case30_governor';
NG  = numel(P0.H);
NB  = size(B.Y_pre, 1);
NC  = 5 * NG + 2 * NB;                                   % channels (90)
gen_cand  = find(B.producing(:) > 0.5);                  % machines that can trip
load_bus  = B.load_bus(:);
line_cand = find(B.br_faultable(:) > 0.5);               % branch indices
[Yr_pre, Vr_pre] = L.kron_reduce(B.Y_pre, B.gen_bus(:), P0.Xdp);
fprintf('%d machines (%d can trip), %d load buses, %d faultable lines\n', ...
        NG, numel(gen_cand), numel(load_bus), numel(line_cand));

%% ---------- 1. random scenario parameters ----------
rng(SEED);
type_id = randi(numel(FAULT_TYPES), N_SCEN, 1);
alpha   = zeros(N_SCEN, 1);
loc     = zeros(N_SCEN, 1);
for k = 1:N_SCEN
    typ = FAULT_TYPES{type_id(k)};
    r   = RANGE.alpha.(typ);
    alpha(k) = r(1) + (r(2) - r(1)) * rand;
    switch typ
        case 'gen',  loc(k) = gen_cand(randi(numel(gen_cand)));     % machine index
        case 'load', loc(k) = load_bus(randi(numel(load_bus)));     % bus number
        case 'line', loc(k) = line_cand(randi(numel(line_cand)));   % branch index
    end
end
unif    = @(r, n) r(1) + (r(2) - r(1)) * rand(n, 1);
t_fault = round(unif(RANGE.t_fault, N_SCEN) / DT) * DT;
t_clear = t_fault + max(DT, round(unif(RANGE.duration, N_SCEN) / DT) * DT);

%% ---------- 2. storage on disk ----------
DT_SAVE = DT * SAVE_EVERY;
tg  = (0:round(T_END / DT_SAVE))' * DT_SAVE;
nT  = numel(tg);
need = N_SCEN * nT * NC * 4 * (1 + SAVE_CLEAN);
fprintf('Stored samples per scenario: %d (dt = %g s), %d channels\n', nT, DT_SAVE, NC);
fprintf('Dataset size: %.1f GB\n', need / 1e9);
free = NaN;
try, free = double(java.io.File(pwd).getFreeSpace()); catch, end
if ~isnan(free)
    fprintf('Free space here: %.1f GB\n', free / 1e9);
    if free < 1.05 * need
        error('Not enough disk space in %s (need %.1f GB).', pwd, need / 1e9);
    end
end

if exist(OUT_FILE, 'file'), delete(OUT_FILE); end
created = datestr(now); %#ok<TNOW1,DATST>
save(OUT_FILE, 'created', '-v7.3');
mf = matfile(OUT_FILE, 'Writable', true);
mf.X(N_SCEN, nT, NC) = single(0);
if SAVE_CLEAN, mf.X_clean(N_SCEN, nT, NC) = single(0); end

%% ---------- 3. simulate ----------
if bdIsLoaded(mdl), close_system(mdl, 0); end
load_system(mdl);
set_param(mdl, 'StopTime', num2str(T_END), 'FixedStep', num2str(DT));
tw = find_system(mdl, 'BlockType', 'ToWorkspace');
for i = 1:numel(tw)
    set_param(tw{i}, 'Decimation', num2str(SAVE_EVERY));
    if endsWith(tw{i}, '_noisy')                 % noise is added below instead
        set_param(tw{i}, 'Commented', 'on');
    end
end
save_system(mdl);
P = P0; %#ok<NASGU>

maxdev = nan(N_SCEN, 1);
failed = false(N_SCEN, 1);
sig    = L.noise_vector(NS, NG, NB);

fprintf('\nSimulating %d scenarios (%g s each, dt = %g s)...\n', N_SCEN, T_END, DT);
fprintf('%-18s %7s %10s %10s %9s %10s\n', 'scenarios', 'done', 'elapsed', 'remaining', 'sim/s', 'OOL so far');
hbar = [];
if SHOW_BAR
    try, hbar = waitbar(0, 'Starting...', 'Name', 'case30 dataset'); catch, hbar = []; end
end
t_start = tic;
for b0 = 1:BATCH:N_SCEN
    idx = (b0:min(b0 + BATCH - 1, N_SCEN))';
    in  = repmat(Simulink.SimulationInput(mdl), numel(idx), 1);
    for j = 1:numel(idx)
        k  = idx(j);
        Pk = L.scenario_P(P0, B, Yr_pre, Vr_pre, FAULT_TYPES{type_id(k)}, alpha(k), ...
                          loc(k), t_fault(k), t_clear(k));
        in(j) = in(j).setVariable('P', Pk);
    end
    outs = parsim(in, 'UseFastRestart', 'on', 'ShowProgress', 'off', ...
                  'StopOnError', 'off', 'TransferBaseWorkspaceVariables', 'off');

    Xb = zeros(numel(idx), nT, NC, 'single');
    if SAVE_CLEAN, Xc = Xb; end
    for j = 1:numel(idx)
        k = idx(j);
        o = outs(j);
        if ~isempty(o.ErrorMessage)
            warning('Scenario %d failed: %s', k, o.ErrorMessage);
            failed(k) = true;
            continue;
        end
        S = L.pack_states(o, tg, NG, NB);
        maxdev(k) = max(abs(S(:, 2:2:2 * NG) - 1), [], 'all');
        rs = RandStream('mt19937ar', 'Seed', SEED + k);
        Xb(j, :, :) = single(S + randn(rs, nT, NC) .* sig);
        if SAVE_CLEAN, Xc(j, :, :) = single(S); end
    end
    mf.X(idx(1):idx(end), :, :) = Xb;
    if SAVE_CLEAN, mf.X_clean(idx(1):idx(end), :, :) = Xc; end
    clear in outs Xb Xc

    done = idx(end);
    el   = toc(t_start);
    rate = done / el;
    left = (N_SCEN - done) / rate;
    ok   = ~failed(1:done);
    fprintf('%6d / %-9d %6.1f%% %10s %10s %9.2f %9.1f%%\n', done, N_SCEN, 100 * done / N_SCEN, ...
            L.fmt_time(el), L.fmt_time(left), rate, 100 * mean(maxdev(ok) > OMEGA_LIMIT));
    if nnz(failed(idx)), fprintf('        (%d failed in this batch)\n', nnz(failed(idx))); end
    if ~isempty(hbar) && isvalid(hbar)
        waitbar(done / N_SCEN, hbar, sprintf('%d / %d scenarios  -  %s remaining', ...
                done, N_SCEN, L.fmt_time(left)));
    end
end
set_param(mdl, 'FastRestart', 'off');
for i = 1:numel(tw), set_param(tw{i}, 'Commented', 'off'); end
save_system(mdl);
if ~isempty(hbar) && isvalid(hbar), close(hbar); end
fprintf('Simulation finished in %s\n', L.fmt_time(toc(t_start)));

%% ---------- 4. labels, split, summary ----------
y_ool = maxdev > OMEGA_LIMIT;
rng(SEED + 1);
perm  = randperm(N_SCEN);
split = zeros(N_SCEN, 1, 'int8');
split(perm(1:round(0.8 * N_SCEN)))                       = 1;
split(perm(round(0.8 * N_SCEN) + 1:round(0.9 * N_SCEN))) = 2;
split(perm(round(0.9 * N_SCEN) + 1:end))                 = 3;

fprintf('\n%-6s %8s %12s\n', 'type', 'count', 'OOL frac');
for c = 1:numel(FAULT_TYPES)
    m = type_id == c & ~failed;
    fprintf('%-6s %8d %12.2f\n', FAULT_TYPES{c}, nnz(m), mean(y_ool(m)));
end
fprintf('overall OOL fraction: %.2f   failed: %d\n', mean(y_ool(~failed)), nnz(failed));

%% ---------- 5. labels and metadata ----------
fprintf('Writing labels and metadata to %s ...\n', OUT_FILE);
mf.t            = tg;
mf.dt_solver    = DT;
mf.n_machines   = NG;
mf.n_buses      = NB;
mf.y_type       = int8(type_id);            % 1 = gen, 2 = load, 3 = line
mf.y_t_fault    = t_fault;
mf.y_ool        = y_ool;
mf.type_names   = strjoin(FAULT_TYPES, ',');
mf.state_names  = L.state_names(NG, NB);
mf.meta_alpha   = alpha;
mf.meta_loc     = loc;                      % gen: machine, load: bus, line: branch index
mf.branches     = [B.br_from(:) B.br_to(:)];  % [from to] of every branch (1-indexed)
mf.gen_bus      = B.gen_bus(:);
mf.meta_t_clear = t_clear;
mf.meta_maxdev  = maxdev;
mf.omega_limit  = OMEGA_LIMIT;
mf.noise_std    = sig;
mf.failed       = failed;
mf.split        = split;
fprintf('Saved %s\n', OUT_FILE);
