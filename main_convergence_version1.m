close all; clear all; clc;

P0_dbm = 20;
P0 = 10^(P0_dbm/10) / 1e3;
N = 20; %antenna number
K = 4;  %user number
M = 30; %symol number
L = 16; %RIS element number
SNR_dB = 20;   %dB
SNR_dB_set = [0:2:10];
SNR = 10^(SNR_dB/10);
N0 = 1/SNR;
epsilon = 0.0001;
Num_iter = 1e0;
max_iteration = 100;
rho = 0.2;
objective_notrade_all = []; objective_notrade_all_sum = zeros(1,max_iteration);
objective_tradeoff_all = [];objective_tradeoff_all_sum = zeros(1,max_iteration);
objective_sdr_notrade_all = [];
objective_sdr_tradeoff_all = [];
gamma_notrade = zeros(1, K); rate_notrade = zeros(1, K); rate_notrade_sum = zeros(size(SNR_dB_set)); rate_notrade_all = zeros(size(SNR_dB_set));
gamma_trade = zeros(1, K); rate_trade = zeros(1, K); rate_trade_sum = zeros(size(SNR_dB_set)); rate_trade_all = zeros(size(SNR_dB_set));
gamma0 = zeros(1, K); rate0 = zeros(1, K); rate0_sum = zeros(size(SNR_dB_set)); rate0_all = zeros(size(SNR_dB_set));
gamma0_trade = zeros(1, K); rate0_trade = zeros(1, K); rate0_trade_sum = zeros(size(SNR_dB_set)); rate0_trade_all = zeros(size(SNR_dB_set));

load Rd.mat;
F = chol(Rd);
complexity_manifold = zeros(size(SNR_dB_set));
complexity_sdr = zeros(size(SNR_dB_set));
complexity_manifold_trade = zeros(size(SNR_dB_set));
complexity_sdr_trade = zeros(size(SNR_dB_set));

for n_iter = 1: Num_iter
    i_snr = 1;
    S = randi(2,K,M)*2 - 3;
    H_bu = (randn(K,N) + 1j* randn(K,N))/sqrt(2);
    H_br = (randn(L,N) + 1j* randn(L,N))/sqrt(2);
    H_ru = (randn(K,L) + 1j* randn(K,L))/sqrt(2);
    nn = 0;
    for SNR_dB = SNR_dB_set
        SNR = 10^(SNR_dB/10);
        N0 = 1/SNR;
        nn = nn+1;
        %% benchmark for given beampattern
        FHS0 = F'*H_bu'*S;
        [U0, S0, V0] = svd(FHS0);
        X0 = sqrt(M) * F* U0* eye(N,M)* V0';
        objective_initial = norm(H_bu*X0 - S, 'fro')^2;
        
        THETA = diag(ones(1,L));
        H = H_bu + H_ru*THETA*H_br;
        H_sdr = H_bu + H_ru*THETA*H_br;
        %% alternating optimization for given beampattern
        delta = 100;
        objective_old = 1e4;
        n_ite = 1;
        objective_notrade_all = [];
        while (delta > epsilon) && (n_ite <= max_iteration-5)
            n_ite = n_ite + 1;
            FHS = F'*H'*S;
            [U1, S1, V1] = svd(FHS);
            X = sqrt(M) *F* U1 * eye(N, M) * V1';
            objective = norm(H*X - S, 'fro')^2;
            %         for i = 1 : K
            %             MUI(i) = mean(abs(H(i,:)*X - S(i, :)).^2);
            %             gamma(i) = 1/(MUI(i)+N0);
            %         end
            %%
            B = H_ru' * H_ru;
            C = H_br*X*X'*H_br';
            T = H_bu * X - S;
            D = H_br*X*T'*H_ru;
            d = diag(D);
            B_C = B.*C.';
            
            tic;
            manifold = complexcirclefactory(L);
            problem.M = manifold;
            problem.cost = @(x) x'*B_C*x + d.'*x + x'*conj(d);
            problem.egrad = @(x) 2*B_C*x + 2*conj(d);
%             checkgradient(problem);
            options.verbosity = 0;
            [x, xcost, info, options] = steepestdescent(problem, [], options);
            
            THETA = diag(x);
            H = H_bu + H_ru*THETA*H_br;
            %         for i = 1 : K
            %             MUI(i) = mean(abs(H(i,:)*X - S(i, :)).^2);
            %             gamma1(i) = 1/(MUI(i)+N0);
            %         end
            objective_new = norm(H*X - S, 'fro')^2;
            delta = objective_old - objective_new;
            objective_old = objective_new;
            objective_notrade_all = [objective_notrade_all, objective_new];
            complexity_manifold(nn) = complexity_manifold(nn) + toc;
            
            tic;
            FHS_sdr = F'*H_sdr'*S;
            [U1, S1, V1] = svd(FHS_sdr);
            X_sdr = sqrt(M) *F* U1 * eye(N, M) * V1';
            B = H_ru' * H_ru;
            C = H_br*X_sdr*X_sdr'*H_br';
            T = H_bu * X_sdr - S;
            D = H_br*X_sdr*T'*H_ru;
            d = diag(D);
            B_C = B.*C.';
            R = [B_C, conj(d); d.', 0];
            cvx_solver sedumi
            cvx_begin quiet
            variable Theta_sdr(L+1, L+1) hermitian semidefinite
            minimize real(trace(R*Theta_sdr));
            subject to
                Theta_sdr(L+1, L+1) == 1;
            cvx_end
            [U,S1] = eig(Theta_sdr);
            theta1 = U(:, L+1);
            Theta_SDR = diag(theta1(1:L));
            H_sdr = H_bu + H_ru*Theta_SDR*H_br;
            objective_sdr_new = norm(H_sdr*X_sdr - S, 'fro')^2;
            objective_sdr_notrade_all = [objective_sdr_notrade_all, objective_sdr_new];
            complexity_sdr(nn) = complexity_sdr(nn) + toc;
        end
        H_notrade = H; X_notrade = X;
        objective_notrade_all = [objective_initial, objective_notrade_all, objective_notrade_all(end)*ones(1, max_iteration-1 - length(objective_notrade_all))];
        objective_sdr_notrade_all = [objective_initial, objective_sdr_notrade_all];
        %% alternating for tradeoff
        delta = 100;
        objective_old = 1e4;
        n_ite = 1;
        U0 = X0;
        b = 100;
        THETA = eye(L,L);
%         THETA = zeros(L,L)
        H = H_bu + H_ru*THETA*H_br;
        H_sdr = H_bu + H_ru*THETA*H_br;
        objective_tradeoff_all = [];
        while (delta > epsilon) && (n_ite <= max_iteration)
            n_ite = n_ite + 1;
            tic;
            %% update X
            A = [sqrt(rho)*H; sqrt(1-rho)*eye(N)];
            B = [sqrt(rho)*S; sqrt(1-rho)*U0];
            Q = A'*A;   G = A'*B;
            [V_Q, D_Q] = eig(Q);
            lambda = diag(D_Q);
            lambda_min = min(lambda);
            u1 = -lambda_min;    u2 = b;
            while abs(u2-u1) > 1e-4
                u3 = (u2+u1)/2;
                u13 = (u1 + u3) / 2;
                u23 = (u2 + u3) / 2;
                P_u3 = norm(V_Q * inv(D_Q+u3*eye(N)) *V_Q' *G, 'fro')^2;
                if P_u3 > M*P0
                    u1 = u3;
                else
                    u2 = u3;
                end
            end
            lambda_opt = u3;
            X_opt = pinv(Q+lambda_opt*eye(N)) * G;
%             trace(X_opt* X_opt')
            
            %% update U
            FX = F'*X_opt;
            [U_bar, S_bar, V_bar] = svd(FX);
            U = sqrt(M)* F * U_bar * eye(N,M) * V_bar';
            
            %% update H
            B1 = H_ru' * H_ru;
            C = H_br*X_opt*X_opt'*H_br';
            T = H_bu * X_opt - S;
            D = H_br*X_opt*T'*H_ru;
            d = diag(D);
            B_C = B1.*C.';
            
            manifold = complexcirclefactory(L);
            problem.M = manifold;
            problem.cost = @(x) x'*B_C*x + d.'*x + x'*conj(d);
            problem.egrad = @(x) 2*B_C*x + 2*conj(d);
%             checkgradient(problem);
            options.verbosity = 0;
            [x, xcost, info, options] = steepestdescent(problem, [], options);
            
            THETA = diag(x);
            H = H_bu + H_ru*THETA*H_br;
%             objective = norm(H*X - S, 'fro')^2;
            objective_new = norm(A*X_opt - B, 'fro')^2;
            delta = objective_old - objective_new;
            objective_old = objective_new;
            objective_tradeoff_all = [objective_tradeoff_all, objective_new];
            complexity_manifold_trade(nn) = complexity_manifold_trade(nn)+toc;
            
            %% sdr-based
            tic;
            %% update X
            A = [sqrt(rho)*H_sdr; sqrt(1-rho)*eye(N)];
            B = [sqrt(rho)*S; sqrt(1-rho)*U0];
            Q = A'*A;   G = A'*B;
            [V_Q, D_Q] = eig(Q);
            lambda = diag(D_Q);
            lambda_min = min(lambda);
            u1 = -lambda_min;    u2 = b;
            while abs(u2-u1) > 1e-4
                u3 = (u2+u1)/2;
                u13 = (u1 + u3) / 2;
                u23 = (u2 + u3) / 2;
                P_u3 = norm(V_Q * inv(D_Q+u3*eye(N)) *V_Q' *G, 'fro')^2;
                if P_u3 > M*P0
                    u1 = u3;
                else
                    u2 = u3;
                end
            end
            lambda_opt = u3;
            X_opt_sdr = pinv(Q+lambda_opt*eye(N)) * G;
%             trace(X_opt* X_opt')
            
            %% update U
            FX = F'*X_opt_sdr;
            [U_bar, S_bar, V_bar] = svd(FX);
            U_sdr = sqrt(M)* F * U_bar * eye(N,M) * V_bar';
            
            %% update H
            B1 = H_ru' * H_ru;
            C = H_br*X_opt_sdr*X_opt_sdr'*H_br';
            T = H_bu * X_opt_sdr - S;
            D = H_br*X_opt_sdr*T'*H_ru;
            d = diag(D);
            B_C = B1.*C.';
            R = [B_C, conj(d); d.', 0];
            cvx_solver sedumi
            cvx_begin quiet
            variable Theta_sdr(L+1, L+1) hermitian semidefinite
            minimize real(trace(R*Theta_sdr));
            subject to
                Theta_sdr(L+1, L+1) == 1;
            cvx_end
            [U,S1] = eig(Theta_sdr);
            theta1 = U(:, L+1);
            Theta_SDR = diag(theta1(1:L));
            H_sdr = H_bu + H_ru*Theta_SDR*H_br;
            objective_sdr_new = norm(H_sdr*X_opt_sdr - S, 'fro')^2;
            objective_sdr_tradeoff_all = [objective_sdr_tradeoff_all, objective_sdr_new];
            complexity_sdr_trade(nn) = complexity_sdr_trade(nn) + toc;
        end
        H_trade = H; X_trade = X_opt;
        objective_tradeoff_all = [objective_initial, objective_tradeoff_all, objective_tradeoff_all(end)*ones(1, max_iteration-1 - length(objective_tradeoff_all))];
        
%         i_snr = i_snr + 1;
    end
%     objective_tradeoff_all_sum = objective_tradeoff_all_sum + objective_tradeoff_all;
%     objective_notrade_all_sum = objective_notrade_all_sum + objective_notrade_all;
%     rate0_all = rate0_all + rate0_sum;
%     rate_notrade_all = rate_notrade_all + rate_notrade_sum;
%     rate0_trade_all = rate0_trade_all + rate0_trade_sum;
%     rate_trade_all = rate_trade_all + rate_trade_sum;
    disp(['Progress - ',num2str(n_iter),'/',num2str(Num_iter)]);
end
% objective_trade_ave = objective_tradeoff_all_sum / Num_iter;
% objective_notrade_ave = objective_notrade_all_sum / Num_iter;
% plot(0:(max_iteration-1), objective_trade_ave, '-b', 'LineWidth', 1.2);
% hold on;
% plot(0:(max_iteration-1), objective_notrade_ave, '-r', 'LineWidth', 1.2);
% grid on;
% xlabel('Iteration number');
% ylabel('Total multi-user interference');
% legend('trade-off', 'strict');
% save('results_convergence_rho_v1.mat','rho', 'L', 'SNR_dB_set', 'objective_trade_ave', 'objective_notrade_ave');
save('results_complexity_v1.mat','rho', 'L', 'SNR_dB_set', 'complexity_manifold', 'complexity_sdr', 'complexity_manifold_trade', 'complexity_sdr_trade');
