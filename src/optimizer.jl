import Knet.Train20: update!, minimize, gclip!, full, gclip_update!, _update!
using LinearAlgebra: norm, lmul!, axpy!

function get_linear_scheduler_with_warmup(max_steps, warmup)
    a = function(t)
        if t<warmup
            return t/warmup
        else
            return 1-((t-warmup)/(max_steps-warmup))
        end
    end
    return a
end

# Adam with weight decay and linear lr scheduler
mutable struct AdamW
    lr::AbstractFloat
    beta1::AbstractFloat
    beta2::AbstractFloat
    eps::AbstractFloat
    t::Int
    gclip::AbstractFloat
    fstm
    scndm
    # A scheduler is simply a function that takes a current time step t and returns a number between
    # 0 and 1 to be multiplied by the lr.
    scheduler
    # wdecayfunc takes a parameter and returns a weightdecay factor
    # If nothing, or 0, no decay happens
    # This is implemented as a function to give more flexibility in deciding which parameters to decay
    wdecayfunc
end

function _update!(w, g, p::AdamW)
    T = eltype(w)
    if p.fstm===nothing; p.fstm=zero(w); p.scndm=zero(w); end
    p.t += 1
    lmul!(p.beta1, p.fstm)
    axpy!(1-p.beta1, g, p.fstm)
    lmul!(p.beta2, p.scndm)
    axpy!(1-p.beta2, g .* g, p.scndm)
    fstm_corrected = p.fstm / T(1 - p.beta1 ^ p.t)
    scndm_corrected = p.scndm / T(1 - p.beta2 ^ p.t)
    lr = p.scheduler != nothing ? p.lr * p.scheduler(p.t) : p.lr 
    wd = p.wdecayfunc != nothing ? p.wdecayfunc(w) : 0 
    axpy!(-lr, (fstm_corrected ./ (sqrt.(scndm_corrected) .+ T(p.eps))) .+ T(wd)*w, w)
end

function gclip_update!(w, g, p::AdamW)
    gclip!(g, p.gclip)          # gclip! supports AutoGrad.Sparse
    g = full(g)
    _update!(w, g, p)
end

AdamW(; lr=0.001, gclip=0, beta1=0.9, beta2=0.999, eps=1e-8, wdecayfunc=nothing, scheduler=nothing)=AdamW(lr, beta1, beta2, eps, 0, gclip, nothing, nothing, scheduler, wdecayfunc)
adamw(f,d;lr=0.001,gclip=0,beta1=0.9,beta2=0.999,eps=1e-8, wdecayfunc=nothing, scheduler=nothing,o...)=minimize(f,d,AdamW(lr,beta1,beta2,eps,0,gclip,nothing,nothing, scheduler, wdecayfunc);o...)
adamw!(x...;o...)=for y in adamw(x...;o...); end

clone(a::AdamW)=AdamW(a.lr,a.beta1,a.beta2,a.eps,0,a.gclip,nothing,nothing, a.scheduler, a.wdecayfunc)

update!(w::AbstractArray{<:Number}, g::AbstractArray, p::AdamW) = gclip_update!(w, g, p)
update!(w::KnetArray, g::KnetArray, p::AdamW) = gclip_update!(w, g, p)

