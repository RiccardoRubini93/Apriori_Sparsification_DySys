using Nabla
using LinearAlgebra
using DelimitedFiles
using NLopt
using Statistics

include("utils.jl")

#define dimension of the start space M and of the landing space N

const M,N = 40,20
#read data from cavity flow

println("System dymension $M x $N")

const Q = reshape(readdlm("Data_2D_2e4/Q.txt"),200,200,200)[1:M,1:M,1:M]
const L = readdlm("Data_2D_2e4/L.txt")[1:M,1:M]
const a = readdlm("Data_2D_2e4/a.txt")[:,1:M]

const eta = -6
#define dimension of the start space M and of the landing space N
#const N_target = reshape(readdlm("N_target"),M,M,M) 
#Q_ = Q[1:N,1:N,1:N]

#indexes = Int32.(readdlm("indexes"))

#m,n = size(indexes)

#for i=1:m
#	Q_[indexes[i,1],indexes[i,2],indexes[i,3]] =  0
#end

#const Q_target   = Q_

#X__ = randn(M,N)
X__ = Matrix{Float64}(I, M, N) + 0.01*randn(M,N)

X_  = X__*(X__'*X__)^(-1/2)

X = reshape(X_,M*N,1)
X = vec(X)

const l1_ori = norm(Q[1:N,1:N,1:N],1)

#compute eigenvalue

const lambda = zeros(M)

for i = 1:N

	lambda[i] = mean(a[:,i].*a[:,i])

end

gamma_ = 1:1:10 # 10 .^(range(-6,stop=-1,length=2))

#prepare some list to store the results
l1_Q   = []
L2_obj = []
obj    = []

opt = Opt(:LD_MMA, M*N)
#opt = Opt(:LD_AUGLAG, M*N)
opt.xtol_rel = 1e-4
opt.maxeval = 100

Xs = []
push!(Xs,X)


for (k,gamma) in enumerate(gamma_)
	
	X    = Xs[k]
	conv = []
	l1   = []
	l2   = []
	println("optimisation for gamma = $k")

	#equality_constraint!(opt,(X,grad)->constraint_L(X,grad),1e-3)
    inequality_constraint!(opt,(X,grad)->constraint_L(X,grad),1e-3)
    inequality_constraint!(opt,(X,grad)->constraint_Q(X,grad,gamma),1e-2)
	min_objective!(opt,(X,grad)->f_L(X,grad,gamma,conv,l1,l2))
	(minf,minx,ret) = optimize(opt, X)
	numevals = opt.numevals

	println("Convergence obtained $ret after $numevals iterations")
	
	writedlm(string("outputs/l1Q_",k),l1)
    writedlm(string("outputs/obj_",k),l2)
	writedlm(string("outputs/conv_",k),conv)
	writedlm(string("outputs/X_",k),minx)
	push!(l1_Q,l1_Q_tilde_to(minx))
	push!(L2_obj,objective(minx))
	push!(obj,minf)
	
	x_ = reshape(minx,M,N)
    xx = x_ #*(x_'*x_)^(-1/2)

	push!(Xs,vec(xx))
end

#writedlm("outputs/conv_history",Conv)
writedlm("outputs/gamma",gamma_)
writedlm("outputs/l1_Q",l1_Q)
writedlm("outputs/L2_obj",L2_obj)
writedlm("outputs/obj",obj)
