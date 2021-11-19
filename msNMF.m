function [wo,ho,fval,RMSE,iter,Wchange,Hchange]=msNMF(Z,s,xvalues,c,f,maxiter,Winit,Hinit,weights,Hprior,type)

% Monotonous slope (ms)NMF: Z=W*H with W being basis functions constrained to be monotonous and have monotonous slope.
% and H being a matrix of mixing coefficients.

% The algorithm is inspired by monotonous (m)NMF by Bhatt Nirav and Arun Ayyar:
% Proceedings of the Second ACM IKDD Conference on Data Sciences. ACM, 2015.

% Developed by Sofie Rahbek. Date: 03/07/20/, revised 09/07/21.

% Associated with publication:
% Rahbek et al. Data-driven separation of MRI signal components for tissue
% characterization. JMR, 2021. Volume 333.
% https://doi.org/10.1016/j.jmr.2021.107103.


% ---------- Inputs -------------
% Z: Data matrix [n X m]

% s: Number of factors

% c: Number of increasing factors out of the total s factors

% f: A sx1 vector defining if slopes are decreasing (-1) OR increasing (1)
% OR a single element accounting for all factors.

% xvalues: the measuring/sample points (e.g. TE-values or b-values) [n x 1].

% maxiter: maximum number of iterations in the ANLS. Default: 1000.

% Winit: initialization of W [n x s]. Default: all values set to 0.1. (Not
% that important, as H will be estimated as the first.)

% Hinit: initialization of H [s x m]. Default: random numbers (uniform)
% between 0 and 1. 

% weights: weighting of the m input signals, [m x 1]. If left empty, no
% weighting is applied (default). 

% Hprior: a possible fixed prior on H for the increasing factors, [c x m].
% If empty, H is not constrained to any prior (default).
% Currently, such prior can only be added to increasing factors.

% type: Choose type of NMF: 'ms' = msNMF (default), 'm' = mNMF, 's'= NMF.

% ----------- Ouputs ------------
% wo: Matrix of s monotonous slope factors [n x s].

% ho: Matrix of mixing coefficients [s x m].

% fval: objective function value for each iteration

% RMSE: final rms-error at convergence. Accounted for possible weighting.

% iter: number of iterations used for convergence

% Wchange: how the W change from iteration to iteration.

% Hchange: how the H change from iteration to iteration.
% ------------------------------

% ---------- Checking number of input parameters -------------:
if nargin < 11
    type = 'ms'; % default
    if nargin < 10
        Hprior = [];
        if nargin < 9
            weights = ones(size(Z,2),1);
            if nargin < 8
                Hinit = [];
                if nargin < 7
                    Winit = [];
                    if nargin < 6
                        maxiter = 1000;
                        if nargin < 5
                            f = ones(s,1); % default: all factors have increasing slope
                            disp('Warning: f not included in input. Set to default: 1 (all factors will have increasing slope).')
                            if nargin < 4
                                c = 0; % default: all factors are decreasing
                                disp('Warning: c not included in input. Set to default: 0 (all factors are decreasing).')
                                if nargin < 3
                                    error('ERROR: too few input arguments')
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


% --------------------------------------------------------

% Constructing matrices for monotonous constraints:
[n,m]=size(Z);
r1=[1 -1 zeros(1,n-2)];
c1=[1;zeros(n-2,1)];
T=toeplitz(c1,r1);
C2=T(1:n-1,:);
I_i=eye(s);
deltaX=diff(xvalues);
C3=[];
for mm=1:n-2
    C3=[C3; zeros(1,mm-1), -1/deltaX(mm), (deltaX(mm)+deltaX(mm+1))/(deltaX(mm)*deltaX(mm+1)), -1/deltaX(mm+1), zeros(1,n-mm-2)];
end

% Vectorizing:
cvec = [ones(numel(1:c),1); (-1)*ones(numel(c+1:s),1)];

if numel(f) == 1
    fvec = f*ones(s,1);
else
    fvec = f;
end

% Constructing decreasing and increasing factors:
Awd = kron(I_i,C2);
crep = reshape(repelem(cvec,(n-1)),[],1); % reshape -> column vector
Awd = crep.*Awd;

% Constructing increasing and decreasing slope of factors:
Awd2 = kron(I_i,C3);
frep = reshape(repelem(fvec,(n-2)),[],1);
Awd2 = frep.*Awd2;

% Defining constraint matrices for W:
if isequal(type,'ms')
    Aw=[Awd;Awd2];
    bw=(10^-8)*ones(size(Aw,1),1);
elseif isequal(type,'m')
    Aw = Awd;
    bw=(10^-8)*ones(size(Aw,1),1);
elseif isequal(type,'s')
    Aw = [];
    bw = [];
else
    error('Error: Do not recognize type')
end

% Check if weights are included correctly:
if numel(weights) ~= m
    weights = ones(m,1);
    disp('Weights are set to one-vector')
end

% Check if data is scaled to start in 1:
if isequal(Z(1,:),ones(1,m))
    scaled = 1;
else
    scaled = 0;
end

% Initialization of while loop
Wold = ones(n,s);
Hold = ones(s,m);

if isempty(Winit)
    W = 0.1*ones(n,s);
else
    W=Winit;
end

if isempty(Hinit)
    H = rand(s,m);
else
    H=Hinit;
end

% the trust-region-reflective algorithm is currently used for estimating H.
options = optimoptions('quadprog','Display','none','Algorithm','trust-region-reflective',... %
    'FunctionTolerance',10^(-16),'MaxIterations',1000); 
% the interior-point algorithm is currenlt used for estimation of W, as the trust-region-reflective algorithm cannot be used when both bounds and equalty constraints are present. 
options1 = optimoptions('quadprog','Display','none','OptimalityTolerance',10^(-16),'MaxIterations',1000); 

iter=0;
tol = 10^-8;

%%% Alternating non-negative least-squares algorithm (ANLS) %%%
while  iter < maxiter && (sum(sum((Hold - H).^2))/m > tol  || sum(sum((Wold - W).^2))/n > tol)
    iter=iter+1;
    
    Wchange(iter)=sum(sum((Wold - W).^2));
    Hchange(iter)=sum(sum((Hold - H).^2));
    
    Wold = W;
    Hold = H;
    
    
    if c > 0 && ~isempty(Hprior)
        % Solving for H matrix while keeping Hprior fixed (for increasing factors):
        Zpart = Z-W(:,1:c)*H(1:c,:);
        Wpart = W(:,c+1:end);
        Qh=2*kron(sparse(diag(weights)),sparse(Wpart'*Wpart));
        fh=-2*Wpart'*Zpart*diag(weights);
        Holdpart = Hold(c+1:end,:);
        LBh = 10^(-8)*ones(size(Holdpart(:)));
        UBh = 20*ones(size(Holdpart(:))) ;
        
        % Estimating H, and then adding the fixed prior H compoonent:
        [h, fval(iter)]= quadprog(Qh,fh(:),[],[],[],[],LBh,UBh,Holdpart(:),options);
        Hpart = myvec2mat(h,size(Holdpart,1));
        H = [Hprior; Hpart]; 
        
    else
        % No prior on H (standard):
        Qh=2*kron(sparse(diag(weights)),sparse(W'*W));
        fh=-2*W'*Z*diag(weights);
        LBh = 10^(-8)*ones(size(H(:))); % lower bound set to 0
        UBh = 20*ones(size(H(:))) ; % upper bound set to 20 (exact value not important, should just be high enough for the given data)
        % Estimating H:
        [h, fval(iter)]= quadprog(Qh,fh(:),[],[],[],[],LBh,UBh,Hold(:),options);
        H = myvec2mat(h,size(H,1));
    end
    % Solving for W matrix:
    Qw=2*kron(H*diag(weights)*H',eye(n));
    fw = -2*Z*diag(weights)*H';
    UBw = ones(size(W))*max(Z(1,:)); % upper bound set to max value in input data.
    LBw = 10^(-8)*ones(size(W)); % lower bound set to 0
    
    % If data is scaled to start in 1, all decreasing components is 
    % constrained to start in 1:
    if scaled==1 
        LBw(1,:)=0.9999999;
    end
    % Growing components should be able to start in 0:
    if c > 0 %
        LBw(1,1:c)= 0;
    end
    
    % Estimating W:
    [w,fval2]= quadprog(Qw,fw(:),Aw,bw,[],[],LBw(:),UBw(:),[],options1);
    W = myvec2mat(w,size(W,1));
    
    
end

% Outputs:
wo = W;
ho = H;
d = Z - wo*ho;
dw = d*diag(weights); % Accounting for weights!
RMSE = sqrt(sum(sum(dw.^2))/(n*m));




