d = load('pinn_dataset.mat');
names = strsplit(d.type_names, ',');
figure;
for c = 1:4
    k = find(d.y_type == c, 1);
    subplot(2,2,c);
    plot(d.t, squeeze(d.X(k,:,2:2:6))); hold on; grid on
    xline(d.y_t_fault(k), 'k--'); xline(d.meta_t_clear(k), 'r--');
    yline(1 + d.omega_limit, ':'); yline(1 - d.omega_limit, ':');
    title(sprintf('%s  \\alpha=%.2f  OOL=%d', names{c}, d.meta_alpha(k), d.y_ool(k)));
    xlabel('t [s]'); ylabel('\omega [pu]')
end