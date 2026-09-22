%% generate_dataset.m
% Generates a randomised fault dataset with the ieee9_governor Simulink model.
%
%   Input  X : [N_SCEN x T x 33] noisy full simulation (Python state layout)
%   Labels   : y_type (fault type), y_t_fault (fault time), y_ool (out-of-limit flag)
%
% Needs: ieee9_gov_simulink_data.mat, ieee9_base.mat (from export_for_simulink.py)
%        and the model built by build_ieee9_gov_model.m (Clock-based version).

clear; clc;

%% ======================== SETTINGS ========================
N_SCEN      = 10000;
T_END       = 8.0;          % simulation length [s] (same for every scenario)
DT          = 1e-3;         % solver step [s]
SAVE_EVERY  = 1;           % store every 10th step -> 0.01 s samples, 801 per scenario
                            % (SAVE_EVERY = 1 stores all 8001 steps: ~10.6 GB file)
BATCH       = 50;           % scenarios per parsim call (progress is printed after each)
SHOW_BAR    = true;         % also show a progress-bar window
SEED        = 42;           % master seed -> the whole dataset is reproducible
OMEGA_LIMIT = 0.005;        % out of limit if |omega - 1| > OMEGA_LIMIT (any gen, any time)
OUT_FILE    = 'pinn_dataset.mat';
SAVE_CLEAN  = false;        % also save noise-free X_clean (doubles the file size)

% Fault types (drawn with equal probability) and uniform ranges
%   gen : generator trip,  Pm0(gen) changes by -alpha * Pm0(gen)
%         (alpha = 0.5 -> half output, alpha = 1 -> zero output)
%   load: extra shunt admittance at a load bus,  Y(bus,bus) += alpha
%   line: line disconnected during the fault (alpha = 1), reconnected at clearing
FAULT_TYPES      = {'gen', 'load', 'line'};
RANGE.alpha.gen  = [0.10 1.00];
RANGE.alpha.load = [0.10 1.00];
RANGE.alpha.line = [1 1];        % always 1 (disconnected); 0 = connected
RANGE.t_fault    = [0.50 4.00];  % fault occurrence time [s]
RANGE.duration   = [0.05 3.00];  % clearing time = t_fault + duration [s]  (<= 7 s,
                                 % leaving >= 1 s of post-clearing response)

% Measurement noise std (same meaning as in build_ieee9_gov_model.m)
NS.delta = 0.01;  NS.omega = 5e-5; NS.Pm = 0.005;
NS.Id    = 0.005; NS.Iq    = 0.005; NS.Vm = 0.001; NS.Th = 0.01;
%% ==========================================================

P0  = load('ieee9_gov_simulink_data.mat');
if ~isfield(P0, 'dPc'), P0.dPc = zeros(3, 1); P0.t_corr = 1e9; P0.t_corr_end = 1e9; end
B   = load('ieee9_base.mat');
mdl = 'ieee9_governor';
gen_bus    = B.gen_bus(:);
load_bus   = B.load_bus(:);
nongen_bus = setdiff((1:size(B.Y_pre, 1))', gen_bus);
% transmission lines = branches between two non-generator buses
% (generator step-up transformers are excluded: opening them islands a machine)
[lr, ls] = find(triu(abs(B.Y_pre), 1) > 1e-9);
keep  = ~ismember(lr, gen_bus) & ~ismember(ls, gen_bus);
lines = [lr(keep) ls(keep)];
fprintf('Transmission lines found: %s\n', mat2str(lines));
[Yr_pre, Vr_pre] = kron_reduce(B.Y_pre, gen_bus, P0.Xdp);

%% ---------- 1. draw all random scenario parameters ----------
rng(SEED);
type_id = randi(numel(FAULT_TYPES), N_SCEN, 1);
alpha   = zeros(N_SCEN, 1);
loc     = zeros(N_SCEN, 1);
for k = 1:N_SCEN
    typ = FAULT_TYPES{type_id(k)};
    r   = RANGE.alpha.(typ);
    alpha(k) = r(1) + (r(2) - r(1)) * rand;
    switch typ
        case 'gen',  loc(k) = randi(numel(gen_bus));              % generator index
        case 'load', loc(k) = load_bus(randi(numel(load_bus)));   % bus number
        case 'line', loc(k) = randi(size(lines, 1));              % row of 'lines'
    end
end
unif    = @(r, n) r(1) + (r(2) - r(1)) * rand(n, 1);
t_fault = round(unif(RANGE.t_fault, N_SCEN) / DT) * DT;           % snapped to the solver grid
t_clear = t_fault + max(DT, round(unif(RANGE.duration, N_SCEN) / DT) * DT);

%% ---------- 2. simulate in batches ----------
DT_SAVE = DT * SAVE_EVERY;
tg  = (0:round(T_END / DT_SAVE))' * DT_SAVE;    % stored time grid
nT  = numel(tg);
fprintf('Stored samples per scenario: %d  (dt = %g s)\n', nT, DT_SAVE);
fprintf('Approx. file size: %.1f GB\n', N_SCEN * nT * 33 * 4 / 1e9 * (1 + SAVE_CLEAN));

% write X to disk batch by batch (the full array may not fit in memory)
if exist(OUT_FILE, 'file'), delete(OUT_FILE); end
created = datestr(now); %#ok<TNOW1,DATST>
save(OUT_FILE, 'created', '-v7.3');
mf = matfile(OUT_FILE, 'Writable', true);
mf.X(N_SCEN, nT, 33) = single(0);                     % preallocate on disk
if SAVE_CLEAN, mf.X_clean(N_SCEN, nT, 33) = single(0); end
maxdev = nan(N_SCEN, 1);
failed = false(N_SCEN, 1);
sig    = noise_vector(NS);

if bdIsLoaded(mdl), close_system(mdl, 0); end   % drop any stale in-memory copy
load_system(mdl);
set_param(mdl, 'StopTime', num2str(T_END), 'FixedStep', num2str(DT));
% log only every SAVE_EVERY-th step (keeps memory small)
tw = find_system(mdl, 'BlockType', 'ToWorkspace');
for i = 1:numel(tw), set_param(tw{i}, 'Decimation', num2str(SAVE_EVERY)); end
save_system(mdl);   % parsim requires a saved model
P = P0; %#ok<NASGU>   default value of P for compilation

fprintf('\nSimulating %d scenarios (%g s each, dt = %g s)...\n', N_SCEN, T_END, DT);
fprintf('%-18s %7s %10s %10s %9s %10s\n', 'scenarios', 'done', 'elapsed', 'remaining', 'sim/s', 'OOL so far');
hbar = [];
if SHOW_BAR
    try, hbar = waitbar(0, 'Starting...', 'Name', 'Generating dataset'); catch, hbar = []; end
end
t_start = tic;
for b0 = 1:BATCH:N_SCEN
    idx = (b0:min(b0 + BATCH - 1, N_SCEN))';
    in  = repmat(Simulink.SimulationInput(mdl), numel(idx), 1);
    for j = 1:numel(idx)
        k  = idx(j);
        Pk = P0;
        Pk.Yred_pre = Yr_pre;   Pk.Vrec_pre = Vr_pre;
        Pk.Yred_post = Yr_pre;  Pk.Vrec_post = Vr_pre;     % original network after clearing
        Pk.dPm0 = zeros(3, 1);
        Yf = B.Y_pre;
        switch FAULT_TYPES{type_id(k)}
            case 'gen'      % generator trip
                Pk.dPm0(loc(k)) = -alpha(k) * P0.Pm0(loc(k));
            case 'load'     % Load_Fault.update(): Y(bus,bus) += alpha
                b = loc(k);
                Yf(b, b) = Yf(b, b) + alpha(k);
            case 'line'     % line open during the fault, reconnected at clearing
                if alpha(k) == 1
                    Yf = open_line_Y(B.Y_pre, lines(loc(k), 1), lines(loc(k), 2));
                end
        end
        [Pk.Yred_fault, Pk.Vrec_fault] = kron_reduce(Yf, gen_bus, P0.Xdp);
        Pk.t_fault = t_fault(k);
        Pk.t_clear = t_clear(k);
        Pk.t_end   = T_END;
        in(j) = in(j).setVariable('P', Pk);
    end

    outs = parsim(in, 'UseFastRestart', 'on', 'ShowProgress', 'off', ...
                  'StopOnError', 'off', 'TransferBaseWorkspaceVariables', 'off');

    Xb = zeros(numel(idx), nT, 33, 'single');
    if SAVE_CLEAN, Xc = Xb; end
    for j = 1:numel(idx)
        k = idx(j);
        o = outs(j);
        if ~isempty(o.ErrorMessage)
            warning('Scenario %d failed: %s', k, o.ErrorMessage);
            failed(k) = true;
            continue;
        end
        S = pack_states(o, tg);                               % [nT x 33] clean
        maxdev(k) = max(abs(S(:, 2:2:6) - 1), [], 'all');
        rs = RandStream('mt19937ar', 'Seed', SEED + k);       % reproducible noise
        Xb(j, :, :) = single(S + randn(rs, nT, 33) .* sig);
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
            fmt_time(el), fmt_time(left), rate, 100 * mean(maxdev(ok) > OMEGA_LIMIT));
    if nnz(failed(idx)), fprintf('        (%d failed in this batch)\n', nnz(failed(idx))); end
    if ~isempty(hbar) && isvalid(hbar)
        waitbar(done / N_SCEN, hbar, sprintf('%d / %d scenarios  -  %s remaining', ...
                done, N_SCEN, fmt_time(left)));
    end
end
set_param(mdl, 'FastRestart', 'off');
if ~isempty(hbar) && isvalid(hbar), close(hbar); end
fprintf('Simulation finished in %s\n', fmt_time(toc(t_start)));

%% ---------- 3. labels, split, summary ----------
y_ool = maxdev > OMEGA_LIMIT;
rng(SEED + 1);
perm  = randperm(N_SCEN);
split = zeros(N_SCEN, 1, 'int8');                 % 1 = train, 2 = val, 3 = test
split(perm(1:round(0.8 * N_SCEN)))                    = 1;
split(perm(round(0.8 * N_SCEN) + 1:round(0.9 * N_SCEN))) = 2;
split(perm(round(0.9 * N_SCEN) + 1:end))              = 3;

fprintf('\n%-6s %8s %12s\n', 'type', 'count', 'OOL frac');
for c = 1:numel(FAULT_TYPES)
    m = type_id == c & ~failed;
    fprintf('%-6s %8d %12.2f\n', FAULT_TYPES{c}, nnz(m), mean(y_ool(m)));
end
fprintf('overall OOL fraction: %.2f   failed: %d\n', mean(y_ool(~failed)), nnz(failed));

%% ---------- 4. save labels and metadata (X is already on disk) ----------
fprintf('Writing labels and metadata to %s ...\n', OUT_FILE);
mf.t           = tg;
mf.dt_solver   = DT;
mf.y_type      = int8(type_id);                   % 1 = gen, 2 = load, 3 = line
mf.y_t_fault   = t_fault;
mf.y_ool       = y_ool;
mf.type_names  = strjoin(FAULT_TYPES, ',');
mf.state_names = ['delta1,omega1,delta2,omega2,delta3,omega3,Pm1,Pm2,Pm3,' ...
                  'Id1,Iq1,Id2,Iq2,Id3,Iq3,' ...
                  strjoin(arrayfun(@(b) sprintf('Vm%d,Th%d', b, b), 1:9, 'uni', 0), ',')];
mf.meta_alpha   = alpha;
mf.meta_loc     = loc;          % gen: generator index, load: bus number, line: row of 'lines'
mf.lines        = lines;        % [from_bus to_bus] of each transmission line
mf.meta_t_clear = t_clear;
mf.meta_maxdev  = maxdev;
mf.omega_limit  = OMEGA_LIMIT;
mf.noise_std    = sig;
mf.failed       = failed;
mf.split        = split;
fprintf('Saved %s\n', OUT_FILE);


%% ======================== helpers ========================
function str = fmt_time(sec)
% seconds -> 'hh:mm:ss'
sec = max(0, round(sec));
str = sprintf('%02d:%02d:%02d', floor(sec / 3600), floor(mod(sec, 3600) / 60), mod(sec, 60));
end

function [Yr, Vr] = kron_reduce(Y, gb, Xdp)
% Reduce the network to generator internal nodes (same as the Python exporter)
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

function Y = open_line_Y(Y, r, s)
% Disconnect line r-s (same as Topology_Change in the Python code;
% the line-charging shunts stay, as they cannot be separated from the Ybus diagonal)
y = -Y(r, s);
Y(r, r) = Y(r, r) - y;  Y(s, s) = Y(s, s) - y;
Y(r, s) = 0;            Y(s, r) = 0;
end

function S = pack_states(o, tg)
% Resample logged signals onto the grid and pack in the Python 33-state layout
g  = @(name) interp1(o.(name).Time, flat(o.(name)), tg);
S  = zeros(numel(tg), 33);
S(:, 1:2:6)   = g('delta');
S(:, 2:2:6)   = g('omega');
S(:, 7:9)     = g('Pm');
S(:, 10:2:15) = g('Id');
S(:, 11:2:15) = g('Iq');
S(:, 16:2:33) = g('Vm');
S(:, 17:2:33) = unwrap(g('Th'));
end

function y = flat(ts)
y = squeeze(ts.Data);
if size(y, 1) ~= numel(ts.Time), y = y.'; end
end

function sig = noise_vector(NS)
sig = zeros(1, 33);
sig(1:2:6)   = NS.delta;  sig(2:2:6)   = NS.omega;  sig(7:9) = NS.Pm;
sig(10:2:15) = NS.Id;     sig(11:2:15) = NS.Iq;
sig(16:2:33) = NS.Vm;     sig(17:2:33) = NS.Th;
end
