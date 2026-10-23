%% compare_with_python.m
% Runs the Simulink model and overlays it on the Python DAE trajectory.
% Python state layout (33 columns): 1:6 delta/omega interleaved, 7:9 Pm,
% 10:15 Id/Iq interleaved, 16:33 Vm/theta interleaved.

P   = load('ieee9_gov_simulink_data.mat');
out = sim('ieee9_governor');
ref = load('python_reference.mat');

% returns [N x nsignals] regardless of how Simulink stored the dimensions
getd = @(ts) reshape(permute(squeeze(ts.Data), ...
    circshift(1:2, double(size(squeeze(ts.Data), 1) ~= numel(ts.Time)))), numel(ts.Time), []);

X = ref.X;  tp = ref.t;
cmp = { 'omega [pu]', out.omega, X(:, 2:2:6)
        'delta [rad]', out.delta, X(:, 1:2:5)
        'Pm [pu]',    out.Pm,    X(:, 7:9)
        'Vm gen buses [pu]', out.Vm, X(:, 16:2:21) };

figure('Name', 'Simulink vs Python DAE');
for k = 1:size(cmp, 1)
    ts = cmp{k, 2};  ys = getd(ts);
    if k == 4, ys = ys(:, 1:3); end        % generator buses 1-3
    subplot(2, 2, k); hold on; grid on;
    plot(ts.Time, ys, '-', 'LineWidth', 1.2);
    set(gca, 'ColorOrderIndex', 1);
    plot(tp, cmp{k, 3}, '--', 'LineWidth', 1.2);
    title(cmp{k, 1}); xlabel('t [s]');
    ysi = interp1(ts.Time, ys, tp);
    fprintf('%-20s max abs diff = %.3e\n', cmp{k, 1}, max(abs(ysi - cmp{k, 3}), [], 'all'));
end
legend({'G1 Simulink', 'G2 Simulink', 'G3 Simulink', 'G1 Python', 'G2 Python', 'G3 Python'});
% Note: Simulink theta is wrapped to (-pi, pi]; use unwrap() before comparing it.
% Note: interp1 at the switching instants picks one side of the jump, so small
% spikes in the algebraic signals (Vm, Id, Iq) at t_fault/t_clear are expected.
