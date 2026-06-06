clear all; close all; clc;

% Load and separate RGB channels
img = imread('peppers.png');                                    % Load peppers.png RGB image
R = double(img(:,:,1));                                         % Red channel as 2D signal
G = double(img(:,:,2));                                         % Green channel as 2D signal
B = double(img(:,:,3));                                         % Blue channel as 2D signal
[M,N] = size(R);                                                % Image dimensions M×N
center = round([M/2, N/2]);                                     % Center location for DC

% ANALYSIS 1: Transformation (2D DFT)
F_R = fftshift(fft2(R));                                        % 2D FFT red, DC centered
F_G = fftshift(fft2(G));                                        % 2D FFT green, DC centered
F_B = fftshift(fft2(B));                                        % 2D FFT blue, DC centered

% ANALYSIS 2: Magnitude Spectrum
mag_R = abs(F_R);                                               % Magnitude spectrum red |F_R(u,v)|
mag_G = abs(F_G);                                               % Magnitude spectrum green |F_G(u,v)|
mag_B = abs(F_B);                                               % Magnitude spectrum blue |F_B(u,v)|
log_mag_R = log(1+mag_R);                                       % Log scale for visualization red
log_mag_G = log(1+mag_G);                                       % Log scale for visualization green
log_mag_B = log(1+mag_B);                                       % Log scale for visualization blue

% ANALYSIS 3: DC Component
DC_R = F_R(center(1),center(2));                                % DC value red F_R(0,0)
DC_G = F_G(center(1),center(2));                                % DC value green F_G(0,0)
DC_B = F_B(center(1),center(2));                                % DC value blue F_B(0,0)
avg_R = abs(DC_R)/(M*N);                                        % Average brightness red
avg_G = abs(DC_G)/(M*N);                                        % Average brightness green
avg_B = abs(DC_B)/(M*N);                                        % Average brightness blue
fprintf('DC Component - R: %.2f, G: %.2f, B: %.2f\n', abs(DC_R), abs(DC_G), abs(DC_B));
fprintf('Avg Brightness - R: %.2f, G: %.2f, B: %.2f\n', avg_R, avg_G, avg_B);

% ANALYSIS 4: Energy Distribution
power_R = mag_R.^2;                                             % Power spectrum red P_R(u,v)=|F_R|²
power_G = mag_G.^2;                                             % Power spectrum green P_G(u,v)=|F_G|²
power_B = mag_B.^2;                                             % Power spectrum blue P_B(u,v)=|F_B|²
[U,V] = meshgrid(1:N,1:M);                                      % Frequency coordinate grid
D = sqrt((U-center(2)).^2+(V-center(1)).^2);                    % Radial distance D(u,v) from DC
max_r = floor(min(M,N)/2);                                      % Maximum radius to analyze
rad_R = zeros(1,max_r);                                         % Radial profile red storage
rad_G = zeros(1,max_r);                                         % Radial profile green storage
rad_B = zeros(1,max_r);                                         % Radial profile blue storage
for r=1:max_r                                                   % Loop through each radius
    mask=(D>=r-0.5)&(D<r+0.5);                                  % Circular ring mask at radius r
    rad_R(r)=mean(power_R(mask));                               % Average power at radius r (red)
    rad_G(r)=mean(power_G(mask));                               % Average power at radius r (green)
    rad_B(r)=mean(power_B(mask));                               % Average power at radius r (blue)
end
cum_R = cumsum(rad_R.^2)/sum(rad_R.^2)*100;                     % Cumulative energy % red
cum_G = cumsum(rad_G.^2)/sum(rad_G.^2)*100;                     % Cumulative energy % green
cum_B = cumsum(rad_B.^2)/sum(rad_B.^2)*100;                     % Cumulative energy % blue
idx90_R = find(cum_R>=90,1);                                    % Radius containing 90% energy red
idx90_G = find(cum_G>=90,1);                                    % Radius containing 90% energy green
idx90_B = find(cum_B>=90,1);                                    % Radius containing 90% energy blue
fprintf('90%% Energy Radius - R: %d, G: %d, B: %d\n', idx90_R, idx90_G, idx90_B);

% ANALYSIS 5: Filtering Operations
D0=30;                                                          % Cutoff frequency D₀
H_LP=exp(-D.^2/(2*D0^2));                                       % Gaussian low-pass H_LP(u,v)
H_HP=1-H_LP;                                                    % Gaussian high-pass H_HP(u,v)
G_LP_R = F_R.*H_LP;                                             % Filtered spectrum red LP: G=F·H
G_HP_R = F_R.*H_HP;                                             % Filtered spectrum red HP: G=F·H
img_LP_R = real(ifft2(ifftshift(G_LP_R)));                      % Inverse FFT red LP → spatial
img_HP_R = real(ifft2(ifftshift(G_HP_R)));                      % Inverse FFT red HP → spatial
G_LP_G = F_G.*H_LP;                                             % Filtered spectrum green LP
G_LP_B = F_B.*H_LP;                                             % Filtered spectrum blue LP
img_LP_G = real(ifft2(ifftshift(G_LP_G)));                      % Inverse FFT green LP
img_LP_B = real(ifft2(ifftshift(G_LP_B)));                      % Inverse FFT blue LP
img_LP = cat(3,uint8(img_LP_R),uint8(img_LP_G),uint8(img_LP_B)); % Reconstruct RGB low-pass image

% ANALYSIS 6: RGB Channel Comparison
HF_thresh = max_r*0.3;                                          % High freq threshold (30% of max)
HF_mask = D>HF_thresh;                                          % Mask for high frequencies
HF_R = sum(power_R(HF_mask))/sum(power_R(:))*100;               % High-freq energy % red
HF_G = sum(power_G(HF_mask))/sum(power_G(:))*100;               % High-freq energy % green
HF_B = sum(power_B(HF_mask))/sum(power_B(:))*100;               % High-freq energy % blue
fprintf('HF Energy %% - R: %.2f, G: %.2f, B: %.2f\n', HF_R, HF_G, HF_B);

% VISUALIZATION - All 6 Analyses Summary
figure('Position', [100,100,1400,900]);                         % Create large figure window

subplot(3,4,1); imshow(img); title('Original peppers.png RGB'); % Show original RGB image
subplot(3,4,2); imshow(uint8(R)); title('Red Channel f_R(x,y)'); % Red channel spatial
subplot(3,4,3); imshow(uint8(G)); title('Green Channel f_G(x,y)'); % Green channel spatial
subplot(3,4,4); imshow(uint8(B)); title('Blue Channel f_B(x,y)'); % Blue channel spatial

subplot(3,4,5); imshow(log_mag_R,[]); title('Red |F_R(u,v)| Spectrum'); colormap(gca,jet); colorbar; % Red magnitude spectrum
subplot(3,4,6); imshow(log_mag_G,[]); title('Green |F_G(u,v)| Spectrum'); colormap(gca,jet); colorbar; % Green magnitude spectrum
subplot(3,4,7); imshow(log_mag_B,[]); title('Blue |F_B(u,v)| Spectrum'); colormap(gca,jet); colorbar; % Blue magnitude spectrum
subplot(3,4,8); bar([avg_R,avg_G,avg_B]); title('DC Component F(0,0)'); set(gca,'XTickLabel',{'R','G','B'}); ylabel('Avg Brightness'); grid on; % DC comparison

subplot(3,4,9); plot(rad_R,'r','LineWidth',2); hold on; plot(rad_G,'g','LineWidth',2); plot(rad_B,'b','LineWidth',2); title('Radial Power Profile P(r)'); xlabel('Radius r'); ylabel('Power'); legend('R','G','B'); grid on; % Radial profiles
subplot(3,4,10); plot(cum_R,'r','LineWidth',2); hold on; plot(cum_G,'g','LineWidth',2); plot(cum_B,'b','LineWidth',2); title('Cumulative Energy'); xlabel('Radius'); ylabel('Energy %'); legend('R','G','B'); grid on; ylim([0 100]); % Cumulative energy

subplot(3,4,11); imshow(img_LP); title('Low-Pass Filtered (Smooth)'); % Smoothed RGB result
subplot(3,4,12); imshow(mat2gray(img_HP_R)); title('High-Pass Filtered (Edges)'); % Edge detected result

sgtitle('Frequency Domain Analysis of peppers.png - Six Analyses Complete', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\n========================================\n');
fprintf('ALL 6 ANALYSES COMPLETE FOR PEPPERS.PNG\n');
fprintf('========================================\n');
fprintf('Image: peppers.png (%d×%d pixels, RGB)\n', M, N);
fprintf('Analysis 1: Transformation ✓\n');
fprintf('Analysis 2: Magnitude Spectrum ✓\n');
fprintf('Analysis 3: DC Component ✓\n');
fprintf('Analysis 4: Energy Distribution ✓\n');
fprintf('Analysis 5: Filtering Operations ✓\n');
fprintf('Analysis 6: RGB Comparison ✓\n');
fprintf('========================================\n');