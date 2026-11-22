function chan_check_sl()
% Simulink channel cross-check for gen_dataset_v2.py: one line fault (line 1, 1.0 -> 1.5 s), S18 gains, v1.3.
here = fileparts(mfilename('fullpath'));  b30 = fullfile(here, '..', '..', 'bus_30');
addpath(b30);  L = case30_lib();
P0 = load(fullfile(b30, 'case30_simulink_data.mat'));  B = load(fullfile(b30, 'case30_base.mat'));
NG = numel(P0.H);  NB = size(B.Y_pre, 1);  [Yr, Vr] = L.kron_reduce(B.Y_pre, B.gen_bus(:), P0.Xdp);
g = load(fullfile(here, 'ceiling_v13x.mat')).G;  G = repmat(reshape(g, 1, NG, 3), 160, 1, 1);
P = L.scenario_P(P0, B, Yr, Vr, 'line', 1, 1, 1.0, 1.5);
C = struct('G', G, 'Pm0', P0.Pm0(:), 'lo', P0.Pm0(:) - 1, 'hi', P0.Pm0(:) + 1, 'rate', Inf, 'h', 1e-3, 'Tl', 0.05, ...
           'Nd', 50, 'trip', zeros(NG, 1), 'freeze', zeros(NG, 1), 'dead', zeros(NG, 1));  %#ok<NASGU>
mdl = 'case30_pidv2';  load_system(fullfile(here, [mdl '.slx']));
o = sim(mdl, 'SrcWorkspace', 'current');
tg = (0:8000)' * 1e-3;
S = L.pack_states(o, tg, NG, NB);
save(fullfile(here, 'chan_check_sl.mat'), 'S');
end
