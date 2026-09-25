%% case30_plot_delta_p_examples.m
% Random scenarios from case30_dataset.mat, each re-simulated without and with
% its corrective delta_p.

clear; clc;
L = case30_lib();

%% ======================== SETTINGS ========================
DATA_FILE   = 'case30_dataset.mat';
N_SHOW      = 4;
PICK_STATUS = 1;       % [] = any; 0 not OOL, 1 corrected, 2 no action, 3 not correctable
PICK_TYPE   = [];      % [] = any; 1 gen, 2 load, 3 line
SEED        = [];
GRACE       = 0.2;
SHOW_PM     = true;
%% ==========================================================

mf      = matfile(DATA_FILE);
y_type  = double(mf.y_type(:, 1));
alpha   = mf.meta_alpha;
loc     = mf.meta_loc;
t_fault = mf.y_t_fault;
t_clear = mf.meta_t_clear;
dp      = mf.y_delta_p;
status  = double(mf.dp_status(:, 1));
t_apply = mf.t_apply;
t_rem   = mf.t_remove;
tg      = mf.t;
DT      = mf.dt_solver;
LIMIT   = mf.omega_limit;
names   = strsplit(mf.type_names, ',');
brs     = mf.branches;
gen_bus = mf.gen_bus;
T_END   = tg(end);

ok = true(size(y_type));
if ~isempty(PICK_STATUS), ok = ok & ismember(status, PICK_STATUS); end
if ~isempty(PICK_TYPE),   ok = ok & ismember(y_type, PICK_TYPE);   end
pool = find(ok);
if isempty(pool), error('No scenario matches the filters.'); end
if ~isempty(SEED), rng(SEED); end
ids = pool(randperm(numel(pool), min(N_SHOW, numel(pool))));
fprintf('Plotting scenarios: %s\n', mat2str(ids(:).'));

P0 = load('case30_simulink_data.mat');
B  = load('case30_base.mat');
NG = numel(P0.H);
[Yr_pre, Vr_pre] = L.kron_reduce(B.Y_pre, B.gen_bus(:), P0.Xdp);

mdl = 'case30_governor';
if bdIsLoaded(mdl), close_system(mdl, 0); end
load_system(mdl);
set_param(mdl, 'StopTime', num2str(T_END), 'FixedStep', num2str(DT));
tw = find_system(mdl, 'BlockType', 'ToWorkspace');
for i = 1:numel(tw), set_param(tw{i}, 'Decimation', '1'); end
save_system(mdl);
P = P0; %#ok<NASGU>

in = repmat(Simulink.SimulationInput(mdl), 2 * numel(ids), 1);
for j = 1:numel(ids)
    k  = ids(j);
    Pk = L.scenario_P(P0, B, Yr_pre, Vr_pre, names{y_type(k)}, alpha(k), loc(k), ...
                      t_fault(k), t_clear(k));
    in(2 * j - 1) = in(2 * j - 1).setVariable('P', Pk);
    Pk.dPc = dp(k, :).';
    Pk.t_corr = t_apply(k);
    Pk.t_corr_end = t_rem(k);
    in(2 * j) = in(2 * j).setVariable('P', Pk);
end
outs = sim(in, 'ShowProgress', 'on');

where = @(k) describe(names{y_type(k)}, loc(k), brs, gen_bus);
nr = 1 + SHOW_PM;
figure('Name', 'case30 delta_p correction', ...
       'Position', [80 80 min(1800, 460 * numel(ids)) 300 * nr + 120]);
fprintf('\n%-6s %-6s %-14s %8s %10s %10s\n', 'id', 'type', 'where', 'status', 'peak no dP', 'peak w/ dP');
gl = arrayfun(@(i) sprintf('G%d', i), 1:NG, 'uni', 0);
for j = 1:numel(ids)
    k  = ids(j);
    o0 = outs(2 * j - 1);  o1 = outs(2 * j);
    t  = o0.omega.Time;
    w0 = L.flat(o0.omega);  w1 = L.flat(o1.omega);
    m  = t >= t_apply(k) + GRACE - 1e-9;
    p0 = max(abs(w0(m, :) - 1), [], 'all');
    p1 = max(abs(w1(m, :) - 1), [], 'all');
    fprintf('%-6d %-6s %-14s %8d %10.5f %10.5f\n', k, names{y_type(k)}, where(k), status(k), p0, p1);

    subplot(nr, numel(ids), j); hold on; grid on;
    plot(t, w0, '-', 'LineWidth', 1.0);
    set(gca, 'ColorOrderIndex', 1);
    plot(t, w1, '--', 'LineWidth', 1.6);
    yline(1 + LIMIT, 'k:'); yline(1 - LIMIT, 'k:');
    xline(t_fault(k), 'k-'); xline(t_clear(k), 'r-'); xline(t_apply(k), 'g-');
    xlabel('t [s]'); ylabel('\omega [pu]');
    dps = strjoin(arrayfun(@(v) sprintf('%.2f', v), dp(k, :), 'uni', 0), ' ');
    title(sprintf('#%d %s %s  \\alpha=%.2f\n\\DeltaP=[%s]  peak %.4f\\rightarrow%.4f', ...
          k, names{y_type(k)}, where(k), alpha(k), dps, p0, p1), 'FontWeight', 'normal');
    if j == 1
        legend([strcat(gl, ' no \DeltaP'), strcat(gl, ' with \DeltaP')], ...
               'Location', 'best', 'FontSize', 6, 'NumColumns', 2);
    end
    if SHOW_PM
        subplot(nr, numel(ids), numel(ids) + j); hold on; grid on;
        plot(t, L.flat(o0.Pm), '-', 'LineWidth', 1.0);
        set(gca, 'ColorOrderIndex', 1);
        plot(t, L.flat(o1.Pm), '--', 'LineWidth', 1.6);
        xline(t_fault(k), 'k-'); xline(t_clear(k), 'r-'); xline(t_apply(k), 'g-');
        xlabel('t [s]'); ylabel('P_m [pu]');
    end
end
sgtitle(sprintf(['solid = no correction, dashed = with \\DeltaP   |   ' ...
                 'black = fault, green = \\DeltaP applied, red = cleared   |   limit \\pm%.3f'], LIMIT));

function s = describe(typ, lc, brs, gen_bus)
switch typ
    case 'gen',  s = sprintf('G%d (bus %d)', lc, gen_bus(lc));
    case 'load', s = sprintf('bus %d', lc);
    case 'line', s = sprintf('line %d-%d', brs(lc, 1), brs(lc, 2));
    otherwise,   s = '';
end
end
