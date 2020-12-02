import Knet.Ops20: elu, relu, selu, sigm, eluback, reluback, seluback, sigmback
import Base.Broadcast: broadcasted
import Knet
using Knet.KnetArrays: KnetArray, DevArray, Bcasted
using Knet.LibKnet8: @knet8
using CUDA: CuArray, CuPtr, @cufunc
using AutoGrad: AutoGrad, @primitive

const GConstant01 = sqrt(2/pi)
const GConstant02 = 0.044715 * sqrt(2/pi)
const GConstant03 = GConstant01 / 2

using CUDA


tanhback(dyi::T,yi::T) where {T<:Number} = dyi*(T(1)-yi*yi)
@primitive tanh(x::DevArray),dy,y tanhback.(dy,y)
@primitive tanhback(dy,y),ddx  ddx.*(1 .- y.*y)  ddx.*(-2 .* dy.*y)


gelu_new(x::T) where T = (x/2)*(1 + tanh(T(GConstant02)*x^3 + T(GConstant01)*x))
gelu_new_back(x::T,dy::T) where T = dy*(T(0.5)*tanh(T(GConstant02)*x^3 + T(GConstant01)*x) + (T(0.0535161)*x^3 + T(GConstant03)*x)*(1/cosh(T(GConstant02)*x^3 + T(GConstant01)*x))^2 + T(0.5))

@primitive  gelu_new(x),dy gelu_new_back.(x,dy)

# This defines gelu for KnetArray
import Base.Broadcast: broadcasted
import Knet: KnetArray

@cufunc gelu_new(x::T) where T = (x/2)*(1 + tanh(T(GConstant02)*x^3 + T(GConstant01)*x))
@cufunc gelu_new_back(x::T,dy::T) where T = dy*(T(0.5)*tanh(T(GConstant02)*x^3 + T(GConstant01)*x) + (T(0.0535161)*x^3 + T(GConstant03)*x)*(1/cosh(T(GConstant02)*x^3 + T(GConstant01)*x))^2 + T(0.5))

function KnetArray(x::CuArray{T,N}) where {T,N}
    p = Base.bitcast(Knet.Cptr, pointer(x))
    k = Knet.KnetPtr(p, sizeof(x), Int(device().handle), x) 
    KnetArray{T,N}(k, size(x))
end

broadcasted(::typeof(gelu_new),x::KnetArray) = KnetArray(gelu_new.(CuArray(x)))
broadcasted(::typeof(gelu_new_back),x::KnetArray,dy::KnetArray) = KnetArray(gelu_new_back.(CuArray(x),CuArray(dy)))

gelu_new