close all; clear all; clc;

P0_dbm = 20;
P0 = 10^(P0_dbm/10) / 1e3;
N = 20; %antenna number
K = 4;  %user number
M = 20; %symol number
L = 16; %RIS element number
SNR_dB = 6;   %dB
delta_L = 4;
L_set = [4:delta_L:32];
SNR = 10^(SNR_dB/10);
N0 = 1/SNR;
epsilon = 0.01;
Num_iter = 1e2;
max_iteration = 100;
rho = 0.01;
objective_notrade_all = [];
objective_tradeoff_all = [];
gamma_notrade = zeros(1, K); rate_notrade = zeros(1, K); rate_notrade_sum = zeros(size(L_set)); rate_notrade_all = zeros(size(L_set));
gamma_trade = zeros(1, K); rate_trade = zeros(1, K); rate_trade_sum = zeros(size(L_set)); rate_trade_all = zeros(size(L_set));
gamma0 = zeros(1, K); rate0 = zeros(1, K); rate0_sum = 0; rate0_all = 0;
gamma0_trade = zeros(1, K); rate0_trade = zeros(1, K); rate0_trade_sum = 0; rate0_trade_all = 0;

load Rd.mat;
F = chol(Rd);

for n_iter = 1: Num_iter
    i_snr = 1;
    S = randi(2,K,M)*2 - 3;
    H_bu = (randn(K,N) + 1j* randn(K,N))/sqrt(2);
    
    %% benchmark for given beampattern
    FHS0 = F'*H_bu'*S;
    [U0, S0, V0] = svd(FHS0);
    X0 = sqrt(M) * F* U0* eye(N,M)* V0';
    
    %% benchmark for tradeoff
    delta = 100;
    objective_old = 1e4;
    n_ite = 1;
    b = 100;
    U = X0;
    while (delta > epsilon) && (n_ite <= max_iteration)
        n_ite = n_ite + 1;
        %% update X
        A = [sqrt(rho)*H_bu; sqrt(1-rho)*eye(N)];
        B = [sqrt(rho)*S; sqrt(1-rho)*U];
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
        
        %% update U
        FX = F'*X_opt;
        [U_bar, S_bar, V_bar] = svd(FX);
        U = sqrt(M)* F * U_bar * eye(N,M) * V_bar';
        
        objective_new = norm(A*X_opt - B, 'fro')^2;
        delta = objective_old - objective_new;
        objective_old = objective_new;
    end
    X0_trade = X_opt;
    
    for i_L = 1:length(L_set)
        L = L_set(i_L);
        
        if i_L == 1;
            H_br = (randn(L,N) + 1j* randn(L,N))/sqrt(2);
            H_ru = (randn(K,L) + 1j* randn(K,L))/sqrt(2);
        else
            H_br = [H_br; (randn(delta_L, N) + 1j*randn(delta_L, N))/sqrt(2)];
            H_ru = [H_ru, (randn(K, delta_L) + 1j*randn(K, delta_L))/sqrt(2)];
        end
        
        THETA = diag(ones(1,L));
        H = H_bu + H_ru*THETA*H_br;
        
        %% alternating optimization for given beampattern
        delta = 100;
        objective_old = 1e4;
        n_ite = 1;
        while (delta > epsilon) && (n_ite <= max_iteration)
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
        end
        H_notrade = H; X_notrade = X;
        
        %% alternating for tradeoff
        delta = 100;
        objective_old = 1e4;
        n_ite = 1;
        U = X0;
        b = 100;
        THETA = eye(L,L);
        H = H_bu + H_ru*THETA*H_br;
        while (delta > epsilon) && (n_ite <= max_iteration)
            n_ite = n_ite + 1;
            %% update X
            A = [sqrt(rho)*H; sqrt(1-rho)*eye(N)];
            B = [sqrt(rho)*S; sqrt(1-rho)*U];
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
        end
        H_trade = H; X_trade = X_opt;
        
        for i = 1 : K
            MUI(i) = mean(abs(H_notrade(i,:)*X_notrade - S(i, :)).^2);
            gamma_notrade(i) = 1/(MUI(i)+N0);
            rate_notrade(i) = log2(1 + gamma_notrade(i));
        end
        rate_notrade_sum(i_L) = sum(rate_notrade);
        for i = 1 : K
            MUI(i) = mean(abs(H_trade(i,:)*X_trade - S(i, :)).^2);
            gamma_trade(i) = 1/(MUI(i)+N0);
            rate_trade(i) = log2(1 + gamma_trade(i));
        end
        rate_trade_sum(i_L) = sum(rate_trade);
    end
    for i = 1 : K
        MUI0(i) = mean(abs(H_bu(i,:)*X0 - S(i, :)).^2);
        gamma0(i) = 1/(MUI0(i)+N0);
        rate0(i) = log2(1 + gamma0(i));
    end
    rate0_sum = sum(rate0);
    for i = 1 : K
        MUI0(i) = mean(abs(H_bu(i,:)*X0_trade - S(i, :)).^2);
        gamma0_trade(i) = 1/(MUI0(i)+N0);
        rate0_trade(i) = log2(1 + gamma0_trade(i));
    end
    rate0_trade_sum = sum(rate0_trade);
    
    rate0_all = rate0_all + rate0_sum;
    rate_notrade_all = rate_notrade_all + rate_notrade_sum;
    rate0_trade_all = rate0_trade_all + rate0_trade_sum;
    rate_trade_all = rate_trade_all + rate_trade_sum;
    disp(['Progress - ',num2str(n_iter),'/',num2str(Num_iter)]);
end
rate0_ave = rate0_all / Num_iter;
rate_notrade_ave = rate_notrade_all / Num_iter;
rate0_trade_ave = rate0_trade_all / Num_iter;
rate_trade_ave = rate_trade_all / Num_iter;
plot([0, L_set], [rate0_trade_ave,rate_trade_ave], '-r*', 'LineWidth', 1.2);
hold on;
plot([0, L_set], [rate0_ave,rate_notrade_ave], '-bo', 'LineWidth', 1.2);
grid on;
xlabel('Number of RIS elements');
ylabel('Sum Rate(bit/s/Hz)');
legend('trade-off', 'strict');
save('results_L_v2.mat', 'L_set', 'SNR_dB', 'rate_trade_ave', 'rate_notrade_ave', 'rate0_trade_ave', 'rate0_ave')
