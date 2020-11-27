import Knet.Ops20: elu, relu, selu, sigm, eluback, reluback, seluback, sigmback
import Base.Broadcast: broadcasted
import Knet
using Knet.KnetArrays: KnetArray, DevArray, Bcasted
using Knet.LibKnet8: @knet8
using CUDA: CuArray, CuPtr
using AutoGrad: AutoGrad, @primitive

const GConstant01 = sqrt(2/pi)
const GConstant02 = 0.044715 * sqrt(2/pi)
const GConstant03 = GConstant01 / 2


# Specialize tanh gradient for DevArrays here. The others have been declared in Ops20 as generic gradients.
tanhback(dyi::T,yi::T) where {T<:Number} = dyi*(T(1)-yi*yi)
@primitive tanh(x::DevArray),dy,y tanhback.(dy,y)
@primitive tanhback(dy,y),ddx  ddx.*(1 .- y.*y)  ddx.*(-2 .* dy.*y)


gelu_new(x::T) where T = (x/2)*(1 + tanh(T(GConstant02)*x^3 + T(GConstant01)*x))
gelu_new_back(x::T,dy::T) where T = dy*(T(0.5)*tanh(T(GConstant02)*x^3 + T(GConstant01)*x) + (T(0.0535161)*x^3 + T(GConstant03)*x)*(1/cosh(T(GConstant02)*x^3 + T(GConstant01)*x))^2 + T(0.5))

@primitive  gelu_new(x),dy gelu_new_back.(x,dy)

for (R,P) in ((KnetArray,Ptr), (CuArray,CuPtr)), T in (Float32,Float64); S = sizeof(T) * 8
    for f in ("gelu_new",)
        J, F = Symbol(f), "$(f)_$S"; M = which(@__MODULE__,J)
        @eval begin
            function broadcasted(::typeof($J),x::$R{$T})
                y = similar(x)
                @knet8($F,(Cint,$P{$T},$P{$T}),length(y),x,y)
                return y
            end
            # Bcasted methods -- only needed for KnetArray
            ($M).$J(x::Bcasted{<:$R{$T}}) = broadcasted($J, x.value) |> Bcasted
            broadcasted(::typeof($J),x::Bcasted{<:$R{$T}}) = broadcasted($J, x.value) |> Bcasted
        end
    end
    for f in ("gelu_new_back",)
        J, F = Symbol(f), "$(f)_$(S)_11"; M = which(@__MODULE__,J)
        @eval begin
            function broadcasted(::typeof($J),x::$R{$T},y::$R{$T})
                z = similar(x)
                @knet8($F,(Cint,$P{$T},$P{$T},$P{$T}),length(z),x,y,z)
                return z
            end
            # Bcasted methods -- only needed for KnetArray
            ($M).$J(x::Bcasted{<:$R{$T}}, y::Bcasted{<:$R{$T}}) = broadcasted($J, x.value, y.value) |> Bcasted
            ($M).$J(x::$R{$T}, y::Bcasted{<:$R{$T}}) = broadcasted($J, x, y.value) |> Bcasted
            ($M).$J(x::Bcasted{<:$R{$T}}, y::$R{$T}) = broadcasted($J, x.value, y) |> Bcasted
            broadcasted(::typeof($J),x::Bcasted{<:$R{$T}}, y::Bcasted{<:$R{$T}}) = broadcasted($J, x.value, y.value) |> Bcasted
            broadcasted(::typeof($J),x::$R{$T}, y::Bcasted{<:$R{$T}}) = broadcasted($J, x, y.value) |> Bcasted
            broadcasted(::typeof($J),x::Bcasted{<:$R{$T}}, y::$R{$T}) = broadcasted($J, x.value, y) |> Bcasted
        end
    end
end

gelu_new