using LinearAlgebra
using Nabla
using Debugger
using TensorOperations
using Printf
using OMEinsum
using Statistics

#kronecker delta
del(k::Integer,j::Integer) = k == j ? 1 : 0

#soft thresholding
S(z, γ) = abs(z) <= γ ? zero(z) : ifelse(z > 0, z - γ, z + γ)

function f_L(x_::Vector,grad::Vector,gamma,conv,l1,l2)

	#objective function : L2 part + l1 part
	#the function returns the value of the objective and its gradient as vector Nx1

	xx  = reshape(x_,M,N)			
	
	X       = xx*(xx'*xx)^(-1/2)
	sum_lam = sum(lambda)	

	if length(grad)>0
		grad[:] = vec(-2/sum_lam*(Diagonal(lambda)*X)) #+ gamma*d_l1_Q_to(x_) 
	end

	
	lam_tilde     = diag(X'*Diagonal(lambda)*X)
	sum_lam_tilde = sum(lam_tilde)
	 

	obj = (sum_lam - sum_lam_tilde)/sum_lam #+ gamma*l1_Q_tilde_to(vec(X)) 

	#grad_ = norm(grad,2)	

	push!(conv,obj)
	push!(l1,l1_Q_tilde(vec(X)))
	push!(l2,(sum_lam - sum_lam_tilde)/sum_lam)	
	l_1 = l1[end]/l1_ori
	TrL = sum(Diagonal(X'*L*X))	

	println("Obj = $obj | cQ = $l_1  |  cL = $TrL")	
	
	return obj

end

##########################

function constraint_L(x_::Vector,grad::Vector)

	xx  = reshape(x_,M,N)			
	X   = xx*(xx'*xx)^(-1/2)

	if length(grad)>0
		grad[:] = vec((L+L')*X)
	end

	return sum(Diagonal(X'*L*X))-eta
end

##########################


function constraint_Q(x_::Vector,grad::Vector,gamma)


	if length(grad)>0
		grad[:] = d_l1_Q_to(x_)
	end

	return l1_Q_tilde_to(x_) - l1_ori/gamma
end

function grad_f(x_)

	#gradient of the L2 part of the objective function computed analytically

	sum_lam       = sum(lambda)
	X  = reshape(x_,M,N)	
	
	return reshape(-2/sum_lam*(Diagonal(lambda)*X),M,N)
end

#########################
#  set of functions used to compute the gradient of the l1 part.
#  every function has two different implementations, one using tensor packages and the other 
#  the double implementation is for checks only and for computational effficiency comparison
#########################

function l1_Q_tilde(x_)

	#l1 norm of the rotaded Qtilde with the standard approach

    X  = reshape(x_,M,N)
    out    = zero(eltype(Q))
    for i = 1:N, j = 1:N, k = 1:N
        partial = zero(eltype(Q))
        for p = 1:M, q = 1:M, r = 1:M
            @inbounds partial += Q[p, q, r]*X[p, i]*X[q, j]*X[r, k]
        end
        out += abs(partial)
    end
    return out
end


################
# l1 norm of the rotated Qtilde with Tensoroperation packages 

#original function

function l1_Q_tilde_to(x_)
    return sum(abs.(Q_tilde_to(x_))) #-Q_target))
end


function Q_tilde_to(x_)
    xx  = reshape(x_,M,N)
	X   = xx*(xx'*xx)^(-1/2)
    @tensor begin 
        Qt[i,j,k] := Q[p, q, r]*X[p, i]*X[q, j]*X[r, k]
    end
    return Qt
end

################

function d_Q_to(x_)

	#Computation of Q[i,j,k] with respect X[m,n] with tensor packages
	
	xx  = reshape(x_,M,N)
	X   = xx*(xx'*xx)^(-1/2)
	d   = Matrix{Float64}(I, N, N) 
	d_  = Matrix{Float64}(I, M, M)	 

	@ein dQ1[m,n,i,j,k] := Q[p,q,r]*X[p,i]*X[q,j]*d[k,n]*d_[r,m] 
	@ein dQ2[m,n,i,j,k] := Q[p,q,r]*X[p,i]*X[r,k]*d[j,n]*d_[q,m] 	
	@ein dQ3[m,n,i,j,k] := Q[p,q,r]*X[r,k]*X[q,j]*d[i,n]*d_[p,m]	      	

	return dQ1 + dQ2 + dQ3
end	

################

function d_l1_Q_to(x_)

	#evaluation of the derivative of ||Q_tilde[i,j,k]||_1 with respect X[m,n] with the tensor 
	#packages
	#the function returns out[m,n] = dQ_tilde_ijk/dX_mn * Q_tilde_ijk/|Q_tilde_ijk| vectorised

	dQ = d_Q_to(x_)
	Qts = sign.(Q_tilde_to(x_)) #-Q_target)
	
	
	@tensor begin		
		out[m,n] := dQ[m,n,i,j,k]*Qts[i,j,k]	
	end	
	
	return vec(out)

end

################################
function d_Q(x_)
	
	#standar formulation of Q[i,j,k] with respect X[m,n]

	X  = reshape(x_,M,N)
	dQ = zeros(M,N,N,N,N)
	
	for m=1:M,n=1:N	
		for i = 1:N, j = 1:N, k = 1:N
			for p = 1:M, q = 1:M, r = 1:M
				@inbounds dQ[m,n,i,j,k] += Q[p,q,r]*(X[p,i]*X[q,j]*del(k,n)*del(r,m) +
										             X[p,i]*X[r,k]*del(j,n)*del(q,m) +		
									                 X[r,k]*X[q,j]*del(i,n)*del(p,m))
			end
		end
	end
	return dQ
end	

##############################
# auxiliary funcions used for check porpuses
#########################

function objective(x_)

	xx  = reshape(x_,M,N)			
	
	X = xx*(xx'*xx)^(-1/2)
	sum_lam       = sum(lambda)
	lam_tilde     = diag(X'*Diagonal(lambda)*X)
	sum_lam_tilde = sum(lam_tilde)
	
	obj = (sum_lam - sum_lam_tilde)/sum_lam
	

	return obj
end

function grad_constr(X)

	#gradient of the l1 part of the objective function computed with automatic differentiation.
	#Only for checks pourposes

	#return ∇(l1_Q_tilde)(X)[1]
	return ∇(l1_N_tilde)(X)[1]
end

function grad_obj(X)

	return ∇(objective)(X)[1]
end

############## constraint on the energy tensor ###################

function l1_N_tilde(x_)

	#l1 norm of the rotaded Qtilde with the standard approach

    X  = reshape(x_,M,N)
	at = a*X
    out    = zero(eltype(Q))
    for i = 1:N, j = 1:N, k = 1:N
        partial = zero(eltype(Q))
        for p = 1:M, q = 1:M, r = 1:M
            @inbounds partial += Q[p, q, r]*X[p, i]*X[q, j]*X[r, k]*mean(at[:,i].*at[:,j].*at[:,k])
        end
        out += abs(partial)
    end
    return out
end


#############################################

function l1_N_tilde_to(x_)
	return sum(abs.(N_tilde_to(x_)))
end

#####################

function N_tilde_to(x_)
	
	Qt = Q_tilde_to(x_)
	X  = reshape(x_,M,N)
	at = a*X
    Nijk  = zero(Qt)	
	
	for i=1:N,j=1:N,k=1:N
		Nijk[i,j,k] = Qt[i,j,k]*mean(at[:,i].*at[:,j].*at[:,k])
	end	
	
	return Nijk
end


