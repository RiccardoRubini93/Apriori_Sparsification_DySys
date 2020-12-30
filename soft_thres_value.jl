using PyPlot;pygui(true);
using DelimitedFiles
using LinearAlgebra
using Nabla
using Debugger
using TensorOperations
using OMEinsum
using Statistics


function dQ_tilde(x_,dx_)

	dQ = d_Q_to(x_)
	dX  = reshape(dx_,M,N)
		
	@tensor begin		
		out[i,j,k] := dQ[m,n,i,j,k]*dX[m,n]	
	end	
	
	return vec(out)
end


include("utils.jl")

const M,N = 40,20
#read data from cavity flow

println("System dymension $M x $N")

const Q = reshape(readdlm("Data_2D_2e4/Q.txt"),200,200,200)[1:M,1:M,1:M]

g   =   readdlm("outputs/gamma")

XX = zeros(M*N,length(g))

for i = 1:length(g)
   
    XX[:,i] = readdlm(string("outputs/X_",i))
    
end

dQ = []
dN = []

dx = 1e-2

for i = 1:length(g)

	X_ = reshape(XX[:,i],M,N)  #+ dx*ones(M,N)
	X = X_*(X_'*X_)^(-0.5)
	dQt = dQ_tilde(vec(X),vec(dx*ones(M,N)))
	push!(dQ,dQt)	
end

writedlm("outputs/Thrs_matrix_Q",dQ)

