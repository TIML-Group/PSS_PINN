%% case30_pidv2_check.m -- v2 Simulink model: no-fault equilibrium + cross-check scenarios.
% Writes pidv2_check_simulink.mat (omega traces) for check_plant_v2.py to compare
% against plant_v2.py.
clear; clc;
here = fileparts(mfilename('fullpath'));
b30  = fullfile(here, '..', '..', 'bus_30');
addpath(b30);
L  = case30_lib();
P0 = load(fullfile(b30, 'case30_simulink_data.mat'));
B  = load(fullfile(b30, 'case30_base.mat'));
NG = numel(P0.H);
[Yr_pre, Vr_pre] = L.kron_reduce(B.Y_pre, B.gen_bus(:), P0.Xdp);
mdl = 'case30_pidv2';  load_system(fullfile(here, [mdl '.slx']));
nl = 160;
g0 = repmat(reshape([20 5 2], 1, 1, 3), nl, NG, 1);
g1 = g0;  g1(41:60, :, 1) = 40;  g1(41:60, :, 3) = 4;      % time-varying test (2.0-3.0 s)
mkC = @(G, trip) struct('G', G, 'Pm0', P0.Pm0(:), 'lo', P0.Pm0(:) - 1, 'hi', P0.Pm0(:) + 1, ...
    'rate', 2.5, 'h', 1e-3, 'Tl', 0.05, 'Nd', 50, 'trip', trip);
sc = { 'none', 0,   1, 1e9,  1e9;
       'load', 7, 0.5, 1.000, 1.500;
       'line', 5,   1, 1.200, 2.000;
       'gen',  1, 0.6, 1.000, 2.500;
       'load', 21, 0.8, 1.500, 1.650 };
W = {};
for i = 1:size(sc, 1)
    [typ, loc, a, tf, tc] = sc{i, :};
    if strcmp(typ, 'none'), P = P0; P.Yred_fault = Yr_pre; P.Yred_pre = Yr_pre; P.Yred_post = Yr_pre;
        P.Vrec_fault = Vr_pre; P.Vrec_pre = Vr_pre; P.Vrec_post = Vr_pre; P.t_fault = tf; P.t_clear = tc;
    else, P = L.scenario_P(P0, B, Yr_pre, Vr_pre, typ, a, loc, tf, tc); end
    trip = zeros(NG, 1);
    if strcmp(typ, 'gen'), trip(loc) = a * P0.Pm0(loc); end
    G = g0;  if i == 2, G = g1; end
    C = mkC(G, trip);
    o = sim(mdl);
    w = L.flat(o.omega);  pm = L.flat(o.Pm);  pe = L.flat(o.Pe);  I = L.flat(o.Iint);
    W{i} = w;  %#ok<SAGROW>
    fprintf('%-5s loc %2d a %.2f [%.3f %.3f]: peak|w-1| %.3e  end|w-1| %.2e  max|Pm-Pm0| %.3f\n', ...
        typ, loc, a, tf, tc, max(abs(w - 1), [], 'all'), max(abs(w(end, :) - 1)), max(abs(pm - P0.Pm0(:)'), [], 'all'));
    if i == 1
        fprintf('   no-fault: I(0)=[%s]  max|Pm-Pe|=%.2e  max|I-Pm0|=%.2e\n', ...
            num2str(I(1, :), '%.4f '), max(abs(pm - pe), [], 'all'), max(abs(I - P0.Pm0(:)'), [], 'all'));
    end
end
t = o.omega.Time;
save(fullfile(here, 'pidv2_check_simulink.mat'), 'W', 't', 'sc');
