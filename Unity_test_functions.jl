using LinearAlgebra
using TensorOperations
using Printf
using OMEinsum
using Nabla
using Test

###################

del(k::Integer,j::Integer) = k == j ? 1 : 0

###################

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

###################

function l1_Q_tilde_to(x_)
    return sum(abs.(Q_tilde_to(x_)))
end

function Q_tilde_to(x_)
    X  = reshape(x_,M,N)
    @tensor begin 
        Qt[i,j,k] := Q[p, q, r]*X[p, i]*X[q, j]*X[r, k]
    end
    return Qt
end

###################

function d_Q_to(x_)
	
	X  = reshape(x_,M,N)
	#dQ = zeros(M,N,N,N,N)
	d  = Matrix{Float64}(I, N, N) 
	d1 = Matrix{Float64}(I, M, M) 
	#@tensor begin 
	@ein dQ1[m,n,i,j,k] := Q[p,q,r]*X[p,i]*X[q,j]*d[k,n]*d1[r,m] 
	@ein dQ2[m,n,i,j,k] := Q[p,q,r]*X[p,i]*X[r,k]*d[j,n]*d1[q,m] 	
	@ein dQ3[m,n,i,j,k] := Q[p,q,r]*X[r,k]*X[q,j]*d[i,n]*d1[p,m]	      	

	return dQ1 + dQ2 + dQ3
end	

##################

function d_Q(x_)
	
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

function d_l1_Q_to(x_)

	#evaluation of the derivative of ||Q_tilde[i,j,k]||_1 with respect X[m,n] with the tensor 
	#packages
	#the function returns out[m,n] = dQ_tilde_ijk/dX_mn * Q_tilde_ijk/|Q_tilde_ijk| vectorised

	dQ = d_Q_to(x_)
	Qts = sign.(Q_tilde_to(x_))

	@tensor begin		
		out[m,n] := dQ[m,n,i,j,k]*Qts[i,j,k]
	end	
	
	return vec(out)

end

#compute gradient through automatic differentiation

d_l1_Q_AD =  ∇(l1_Q_tilde)

#define constants

const N = 5
const M = 10
const Q = rand(M, M, M);
x_ = randn(M*N);
δ = vec(Matrix{Float64}(I, M, N))

#test that the functions give the expected results

@testset "SID functions" begin 
	@testset "Dimension and type check" begin
		@test size(Q)                  == (M,M,M)
		@test size(Q_tilde_to(x_))     == (N,N,N)
		@test length(d_l1_Q_to(x_))    == M*N
		@test length(d_l1_Q_AD(x_)[1]) == M*N
		@test eltype(d_l1_Q_AD(x_)[1]) == eltype(d_l1_Q_to(x_))
	end
	
	@testset "Basic Functions" begin	
		@testset "Rotations" begin			
			@test l1_Q_tilde(x_)    ≈ l1_Q_tilde_to(x_)
			@test l1_Q_tilde(δ)     ≈ l1_Q_tilde_to(δ)
			@test l1_Q_tilde(x_)    ≈ norm(Q_tilde_to(x_),1)
			@test l1_Q_tilde_to(x_) ≈ norm(Q_tilde_to(x_),1)
			@test l1_Q_tilde(δ)     ≈ norm(Q[1:N,1:N,1:N],1)
			@test l1_Q_tilde_to(δ)  ≈ norm(Q[1:N,1:N,1:N],1)
		end	
	end	
	
	@testset "Composed Functions" begin
		@testset "Gradient Q_tilde" begin		
			@test d_Q_to(x_) ≈ d_Q(x_)
		end	
	end	
	@testset "Check Gradient" begin
		@testset "l1_Gradient" begin		
			@test d_l1_Q_to(x_) ≈ d_l1_Q_AD(x_)[1]
		end	
	end
	@testset "Additonal tests involving the identity" begin
		@test Q_tilde_to(δ) ≈ Q[1:N,1:N,1:N]
		
	end

end
		


	



