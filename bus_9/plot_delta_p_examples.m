%% plot_delta_p_examples.m
% Picks random scenarios from pinn_dataset.mat and re-simulates each one twice
% in Simulink -- without and with its corrective delta_p -- then plots them.
%
% Needs: pinn_dataset.mat (with the delta_p labels), ieee9_gov_simulink_data.mat,
%        ieee9_base.mat, and the model built by build_ieee9_gov_model.m.

clear; clc;

%% ======================== SETTINGS ========================
DATA_FILE   = 'pinn_dataset.mat';
N_SHOW      = 4;            % how many scenarios to plot
PICK_STATUS = 1;            % [] = any; 1 = corrected, 2 = no action needed,
                            % 3 = not correctable, 0 = not OOL
PICK_TYPE   = [];           % [] = any; 1 = gen, 2 = load, 3 = line
SEED        = [];           % [] = different scenarios every run, or a number to repeat
GRACE       = 0.2;          % must match compute_delta_p.m
SHOW_PM     = true;         % second row of plots with the mechanical power
%% ==========================================================

mf      = matfile(DATA_FILE);
y_type  = double(mf.y_type(:, 1));
alpha   = mf.meta_alpha;
loc     = mf.meta_loc;
lines   = mf.lines;
t_fault = mf.y_t_fault;
t_clear = mf.meta_t_clear;
y_ool   = logical(mf.y_ool);
dp      = mf.y_delta_p;
status  = double(mf.dp_status(:, 1));
t_apply = mf.t_apply;
t_rem   = mf.t_remove;
tg      = mf.t;
DT      = mf.dt_solver;
LIMIT   = mf.omega_limit;
names   = strsplit(mf.type_names, ',');
T_END   = tg(end);

%% ---------- pick scenarios ----------
ok = true(size(y_type));
if ~isempty(PICK_STATUS), ok = ok & ismember(status, PICK_STATUS); end
if ~isempty(PICK_TYPE),   ok = ok & ismember(y_type, PICK_TYPE);   end
pool = find(ok);
if isempty(pool), error('No scenario matches the filters.'); end
if ~isempty(SEED), rng(SEED); end
ids = pool(randperm(numel(pool), min(N_SHOW, numel(pool))));
fprintf('Plotting scenarios: %s\n', mat2str(ids(:).'));

%% ---------- simulate each one twice ----------
P0 = load('ieee9_gov_simulink_data.mat');
if ~isfield(P0, 'dPc'), P0.dPc = zeros(3, 1); P0.t_corr = 1e9; P0.t_corr_end = 1e9; end
B  = load('ieee9_base.mat');
[Yr_pre, Vr_pre] = kron_reduce(B.Y_pre, B.gen_bus(:), P0.Xdp);

mdl = 'ieee9_governor';
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
    Pk = scenario_P(P0, B, Yr_pre, Vr_pre, names{y_type(k)}, alpha(k), loc(k), ...
                    lines, t_fault(k), t_clear(k));
    in(2 * j - 1) = in(2 * j - 1).setVariable('P', Pk);      % without correction
    Pk.dPc = dp(k, :).';
    Pk.t_corr = t_apply(k);
    Pk.t_corr_end = t_rem(k);
    in(2 * j) = in(2 * j).setVariable('P', Pk);              % with correction
end
outs = sim(in, 'ShowProgress', 'on');

%% ---------- plot ----------
nr = 1 + SHOW_PM;
figure('Name', 'delta_p correction', 'Position', [80 80 min(1800, 460 * numel(ids)) 300 * nr + 120]);
fprintf('\n%-6s %-6s %8s %8s %8s %10s %10s\n', 'id', 'type', 'status', 't_fault', 't_clear', ...
        'peak no dP', 'peak w/ dP');
for j = 1:numel(ids)
    k  = ids(j);
    o0 = outs(2 * j - 1);  o1 = outs(2 * j);
    t  = o0.omega.Time;
    w0 = flat(o0.omega);   w1 = flat(o1.omega);
    m  = t >= t_apply(k) + GRACE - 1e-9;          % the window delta_p is judged on
    p0 = max(abs(w0(m, :) - 1), [], 'all');
    p1 = max(abs(w1(m, :) - 1), [], 'all');
    fprintf('%-6d %-6s %8d %8.2f %8.2f %10.5f %10.5f\n', k, names{y_type(k)}, status(k), ...
            t_fault(k), t_clear(k), p0, p1);

    subplot(nr, numel(ids), j); hold on; grid on;
    plot(t, w0, '-',  'LineWidth', 1.0);
    set(gca, 'ColorOrderIndex', 1);
    plot(t, w1, '--', 'LineWidth', 1.6);
    yline(1 + LIMIT, 'k:'); yline(1 - LIMIT, 'k:');
    xline(t_fault(k), 'k-');  xline(t_clear(k), 'r-');  xline(t_apply(k), 'g-');
    xlabel('t [s]'); ylabel('\omega [pu]');
    title(sprintf('#%d %s  \\alpha=%.2f  loc=%d\n\\DeltaP=[%.2f %.2f %.2f]  peak %.4f\\rightarrow%.4f', ...
          k, names{y_type(k)}, alpha(k), loc(k), dp(k, 1), dp(k, 2), dp(k, 3), p0, p1), ...
          'FontWeight', 'normal');
    if j == 1
        legend({'G1 no \DeltaP', 'G2 no \DeltaP', 'G3 no \DeltaP', ...
                'G1 with \DeltaP', 'G2 with \DeltaP', 'G3 with \DeltaP'}, ...
                'Location', 'best', 'FontSize', 7);
    end

    if SHOW_PM
        subplot(nr, numel(ids), numel(ids) + j); hold on; grid on;
        plot(t, flat(o0.Pm), '-', 'LineWidth', 1.0);
        set(gca, 'ColorOrderIndex', 1);
        plot(t, flat(o1.Pm), '--', 'LineWidth', 1.6);
        xline(t_fault(k), 'k-');  xline(t_clear(k), 'r-');  xline(t_apply(k), 'g-');
        xlabel('t [s]'); ylabel('P_m [pu]');
    end
end
sgtitle(sprintf(['solid = no correction, dashed = with \\DeltaP   |   ' ...
                 'black = fault, green = \\DeltaP applied, red = cleared   |   limit \\pm%.3f'], LIMIT));


%% ======================== helpers ========================
function Pk = scenario_P(P0, B, Yr_pre, Vr_pre, typ, a, lc, lines, tf, tc)
Pk = P0;
Pk.Yred_pre  = Yr_pre;  Pk.Vrec_pre  = Vr_pre;
Pk.Yred_post = Yr_pre;  Pk.Vrec_post = Vr_pre;
Pk.dPm0 = zeros(3, 1);
Yf = B.Y_pre;
switch typ
    case 'gen',  Pk.dPm0(lc) = -a * P0.Pm0(lc);
    case 'load', Yf(lc, lc) = Yf(lc, lc) + a;
    case 'line', if a == 1, Yf = open_line_Y(B.Y_pre, lines(lc, 1), lines(lc, 2)); end
    otherwise,   error('Unknown fault type "%s"', typ);
end
[Pk.Yred_fault, Pk.Vrec_fault] = kron_reduce(Yf, B.gen_bus(:), P0.Xdp);
Pk.t_fault = tf;
Pk.t_clear = tc;
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
