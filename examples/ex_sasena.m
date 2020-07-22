
% Simulation-based optimization of Sasena test problem using MCSKopt
% By Resul Al @DTU

clc; clearvars;
% Define an optimization problem structure p
p      = struct;
p.lbs  = zeros(1,2);      % Lower bounds of the design space
p.ubs  = ones(1,2);       % Upper bounds of the design space
p.x0   = (p.lbs+p.ubs)/2; 
p.dim  = numel(p.x0);     % Dimensinality of the design space
p.m    = 1e3;             % MC sample size for integrated uncertainty analysis
p.cv   = 0.50;            % Coefficient of variation in uncertain parameters
k      = 5*p.dim;         % Size of initial (coarse) design
Nmax   = 100;              % MaxFunEval (SK iterations) 

% Create an initial (coarse) design space (InitialX)
Xp = lhsdesign(k,p.dim);  % LHS in probability space
for i=1:p.dim, InitialX(:,i) = unifinv(Xp(:,i),p.lbs(i),p.ubs(i)); end 
[InitialObjectiveObservations, InitialConstraintObservations] = rs_simulate(InitialX,p);


% Prepare variables and the objective function for SK optimization
vars=[];
for i=1:p.dim
    eval(sprintf("x%d = optimizableVariable('x%d',[%d,%d]);",i,i,p.lbs(i),p.ubs(i)));
    eval(sprintf("vars = [vars x%d];",i));
end

% Call the interface to the optimizer
fun = @(xx) myObj(xx,p); % objective function
[x,fval,results] = MOSKopt(fun,vars,'Verbose',1,...
                            'SaveEachNiters',100,...
                            'MaxObjectiveEvaluations',Nmax,...
                            'NumSeedPoints',k,...
                            'NumRepetitions',p.m,...
                            'InitialX',array2table(InitialX),...
                            'InitialObjectiveObservations',InitialObjectiveObservations,...
                            'InitialConstraintObservations',InitialConstraintObservations,...
                            'NumCoupledConstraints',3,...
                            'CoupledConstraintTolerances',1e-3*ones(1,3),...
                            'InfillCriterion','mcFEI',... % 'FEI', 'cAEI'
                            'InfillSolver','particleswarm',... % 'GlobalSearch', 'MultiStart'
                            'UncertaintyHedge','Mean') % 'MeanPlusSigma', 'UCI95', 'PF80'

function [f,g,UserData] = myObj(xx,p) 
    % Handle of the objective function that returns the objective and the constraint
    % observations (each with m MC simulations).  
    x=[];
    for i=1:p.dim
        eval(sprintf("x = [x ; xx.x%d];",i))
    end 
    [f_observations, g_observations] = rs_simulate(x',p); % Calls the simulator
    
    % Outputs: Mean of the objective, constraints as well as entire dataset from MC simulations.
    f = nanmean(f_observations,2); 
    g = cellfun(@(X) nanmean(X,2), g_observations ,'UniformOutput',false); 
    UserData.ObjectiveObservations    = f_observations; 
    UserData.ConstraintObservations   = g_observations; 
end


function [f_observations, g_observations] = rs_simulate(X,p)
    % simulates the black-box m times and returns the values of 
    % objective and constraints in row vectors.
    f  = @simulator;
    k  = size(X,1); % # of design points
    d  = size(X,2); % dim
    m  = p.m;
    cv = p.cv; % coefficient of variation
    
    if isfield(p,'seed'), rng(p.seed,'Twister'); end

    Xu = norminv(lhsdesign(m,4),1,cv);  % mean=1, std=cv
    
    
    % Create large matrices for vectorized MCS
    XXu = repmat(Xu,k,1);
    XXd = zeros(k*m,d);
    for i=1:k
        XXd((i-1)*m+1:i*m,:) = repmat(X(i,:),m,1); 
    end
    % perform vectorized MCS
    YYo = f(XXd,XXu);
    
    f_observations  = reshape(YYo(:,1),m,k)';
    g1_observations = reshape(YYo(:,2),m,k)';
    g2_observations = reshape(YYo(:,3),m,k)';
    g3_observations = reshape(YYo(:,4),m,k)';
    g_observations  = {g1_observations, g2_observations, g3_observations};
    
end

function Y = simulator(X,U)
    if nargin==1
        Y = [objective(X) constraints(X)];
    else 
        Y = [objective(X,U) constraints(X,U)];
    end
end

function obj = objective(X,U)
    % vectorized sasena objective
    x1 = X(:,1);
    x2 = X(:,2);
    
    if nargin>1
        u1  = U(:,1);
    else
        u1=1; % nominal values
    end   
    obj = -1.*u1.*(x1-1).^2  - 1.*(x2-0.5).^2;
end

function cons = constraints(X,U)
    % vectorized and parametrized sasena constraints
    x1 = X(:,1);
    x2 = X(:,2);
    
    if nargin>1
        u2  = U(:,2);
        u3  = U(:,3);
        u4  = U(:,4);
    else
        u2=1; u3=1; u4=1; % nominal values
    end
       
    g1   = ( u2.*(x1-3).^2 + (x2+2).^2 ).*exp(-x2.^7) - 12;
    g2   = u3.*10.*x1 + x2 - 7;
    g3   = (x1-0.5).^2 + u4.*(x2-0.5).^2 - 0.2;
    cons = [g1 g2 g3];
end
