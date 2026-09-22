%% plot_noisy.m
% Runs the model, plots clean vs noisy signals, and saves them to a .mat file.

P   = load('ieee9_gov_simulink_data.mat');
out = sim('ieee9_governor');

% returns [N x nsignals] regardless of how Simulink stored the dimensions
getd = @(ts) reshape(permute(squeeze(ts.Data), ...
    circshift(1:2, double(size(squeeze(ts.Data), 1) ~= numel(ts.Time)))), numel(ts.Time), []);

sigs   = {'omega', 'delta', 'Pm', 'Vm'};
titles = {'omega [pu]', 'delta [rad]', 'Pm [pu]', 'Vm gen buses [pu]'};

figure('Name', 'Clean vs noisy');
for k = 1:numel(sigs)
    clean = out.(sigs{k});
    noisy = out.([sigs{k} '_noisy']);
    yc = getd(clean);  yn = getd(noisy);
    if strcmp(sigs{k}, 'Vm'), yc = yc(:, 1:3); yn = yn(:, 1:3); end
    subplot(2, 2, k); hold on; grid on;
    plot(noisy.Time, yn, '.', 'MarkerSize', 6);
    set(gca, 'ColorOrderIndex', 1);
    plot(clean.Time, yc, '-', 'LineWidth', 1.5);
    title(titles{k}); xlabel('t [s]');
end
legend({'G1 noisy', 'G2 noisy', 'G3 noisy', 'G1 clean', 'G2 clean', 'G3 clean'});

% ---- save everything: clean and noisy, one row per time step ----
names = {'delta', 'omega', 'Pm', 'Id', 'Iq', 'Vm', 'Th'};
S.t = out.delta.Time;
for k = 1:numel(names)
    S.(names{k}) = getd(out.(names{k}));
    nn = [names{k} '_noisy'];
    if isprop(out, nn) || any(strcmp(out.who, nn))
        S.(nn) = getd(out.(nn));
    end
end
save('simulink_noisy_data.mat', '-struct', 'S');
fprintf('Saved simulink_noisy_data.mat\n');
