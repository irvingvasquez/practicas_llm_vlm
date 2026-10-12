"""Pruebas de la práctica 03_01: mecanismos de atención.

Cada función recibe el código del estudiante. El notebook sólo la llama.
Los valores esperados se calcularon con pesos fijos (no aleatorios), de modo
que las pruebas no dependen del orden en que se crean las capas.
"""

import torch
from torch import nn

# Embeddings de "Tu viaje empieza con un paso" (los mismos del notebook)
ENTRADAS = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)

_ESPERADO_1 = torch.tensor([
        [ 0.44206,  0.59310,  0.57899],
        [ 0.44187,  0.65148,  0.56831],
        [ 0.44313,  0.64959,  0.56707],
        [ 0.43039,  0.62983,  0.55103],
        [ 0.46710,  0.59099,  0.52660],
        [ 0.41772,  0.65032,  0.56454],
    ])

_ESPERADO_2 = torch.tensor([
        [ 0.90201, -0.64827],
        [ 0.86781, -0.67257],
        [ 0.86799, -0.67189],
        [ 0.84853, -0.69182],
        [ 0.86620, -0.66539],
        [ 0.84935, -0.69585],
    ])

_ESPERADO_4 = torch.tensor([
        [ 1.08000, -0.39000],
        [ 1.09855, -0.63768],
        [ 1.09359, -0.73771],
        [ 0.98652, -0.69003],
        [ 0.82941, -0.72537],
        [ 0.84935, -0.69585],
    ])

_ESPERADO_6 = torch.tensor([
        [ 1.99250, -0.51750,  0.07562,  1.60812],
        [ 2.11020, -1.30097,  0.30734,  1.67081],
        [ 2.13913, -1.61417,  0.40022,  1.68298],
        [ 1.93539, -1.61935,  0.44033,  1.49610],
        [ 1.73098, -1.60743,  0.43285,  1.37006],
        [ 1.63098, -1.53367,  0.20858,  1.29321],
    ])

_SESGO_SALIDA = [0.1, -0.2, 0.3, 0.0]


def _pesos(filas, columnas, semilla):
    """Matriz determinista con valores en [-1.25, 1.25]."""
    return torch.tensor([[((i * 7 + j * 3 + semilla * 5) % 11 - 5) / 4 for j in range(columnas)]
                         for i in range(filas)])


def _fijar(capa, nombre, filas, columnas, semilla, sesgo=None):
    assert isinstance(capa, nn.Linear), f"{nombre} debe ser una capa nn.Linear"
    assert tuple(capa.weight.shape) == (filas, columnas), \
        f"{nombre}.weight debería tener forma {(filas, columnas)} pero tiene {tuple(capa.weight.shape)}"
    with torch.no_grad():
        capa.weight.copy_(_pesos(filas, columnas, semilla))
        if sesgo is not None:
            capa.bias.copy_(torch.tensor(sesgo))


def _fijar_qkv(modulo, d_in, d_out):
    _fijar(modulo.W_query, "W_query", d_out, d_in, 1)
    _fijar(modulo.W_key, "W_key", d_out, d_in, 2)
    _fijar(modulo.W_value, "W_value", d_out, d_in, 3)


def _cerca(obtenido, esperado, mensaje):
    assert isinstance(obtenido, torch.Tensor), f"Se esperaba un tensor pero se obtuvo {type(obtenido)}"
    assert tuple(obtenido.shape) == tuple(esperado.shape), \
        f"{mensaje}: se esperaba forma {tuple(esperado.shape)} pero se obtuvo {tuple(obtenido.shape)}"
    assert torch.allclose(obtenido.detach().float(), esperado, atol=1e-4), \
        f"{mensaje}: los valores no coinciden.\nEsperado:\n{esperado}\nObtenido:\n{obtenido.detach()}"


def prueba_1(atencion_simple):
    resultado = atencion_simple(ENTRADAS)
    assert isinstance(resultado, (tuple, list)) and len(resultado) == 2, \
        "atencion_simple debe regresar (contexto, pesos)"
    contexto, pesos = resultado
    assert isinstance(pesos, torch.Tensor) and tuple(pesos.shape) == (6, 6), \
        "Los pesos de atención deben ser una matriz de (num_tokens, num_tokens)"
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(6), atol=1e-5), "Cada fila de pesos debe sumar 1"
    assert (pesos >= 0).all(), "Los pesos de atención deben ser positivos (usa softmax)"
    _cerca(contexto, _ESPERADO_1, "Vectores de contexto")

    x = torch.randn(4, 5)
    contexto, pesos = atencion_simple(x)
    assert tuple(contexto.shape) == (4, 5), "Debe funcionar con cualquier número de tokens y dimensión"
    assert torch.allclose(contexto, pesos @ x, atol=1e-5), "El contexto debe ser la suma ponderada de las entradas"


def prueba_2(SelfAttention_v2):
    sa = SelfAttention_v2(3, 2)
    assert sa.W_query.bias is None, "Con qkv_bias=False las capas no deben tener sesgo"
    _fijar_qkv(sa, 3, 2)
    _cerca(sa(ENTRADAS), _ESPERADO_2, "Salida de SelfAttention_v2")

    sa = SelfAttention_v2(3, 2, qkv_bias=True)
    assert all(c.bias is not None for c in (sa.W_query, sa.W_key, sa.W_value)), \
        "Con qkv_bias=True las tres capas deben tener sesgo"
    n = sum(p.numel() for p in sa.parameters())
    assert n == 3 * (3 * 2 + 2), f"Se esperaban {3 * (3 * 2 + 2)} parámetros entrenables pero hay {n}"


class _ConPesosLineales(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)


def prueba_3(transferir_pesos, SelfAttention_v1):
    torch.manual_seed(0)
    sa_v1 = SelfAttention_v1(3, 2)
    sa_v2 = _ConPesosLineales(3, 2)
    _fijar_qkv(sa_v2, 3, 2)

    transferir_pesos(sa_v2, sa_v1)

    for nombre in ("W_query", "W_key", "W_value"):
        p = getattr(sa_v1, nombre)
        assert isinstance(p, nn.Parameter), f"sa_v1.{nombre} debe seguir siendo un nn.Parameter"
        assert tuple(p.shape) == (3, 2), f"sa_v1.{nombre} debe conservar la forma (d_in, d_out) = (3, 2)"
    assert len(list(sa_v1.parameters())) == 3, "sa_v1 debe seguir teniendo 3 parámetros registrados"
    _cerca(sa_v1(ENTRADAS), _ESPERADO_2, "Salida de SelfAttention_v1 después de transferir los pesos")


def prueba_4(CausalAttention):
    lote = torch.stack((ENTRADAS, ENTRADAS), dim=0)
    ca = CausalAttention(3, 2, context_length=6, dropout=0.0)
    assert "mask" in dict(ca.named_buffers()), "La máscara debe registrarse con self.register_buffer('mask', ...)"
    _fijar_qkv(ca, 3, 2)
    _cerca(ca(lote), torch.stack((_ESPERADO_4, _ESPERADO_4)), "Salida de CausalAttention")

    ca = CausalAttention(3, 2, context_length=10, dropout=0.0)
    _fijar_qkv(ca, 3, 2)
    _cerca(ca(lote), torch.stack((_ESPERADO_4, _ESPERADO_4)),
           "Con context_length=10 y entradas de 6 tokens (recorta la máscara a num_tokens)")

    otro = lote.clone()
    otro[:, -1] = torch.tensor([5.0, -3.0, 2.0])
    salida = ca(otro)
    assert torch.allclose(salida[:, :-1], ca(lote)[:, :-1], atol=1e-6), \
        "Cambiar el último token no debe afectar a los anteriores: la atención debe ser causal"

    ca = CausalAttention(3, 2, context_length=6, dropout=0.5)
    _fijar_qkv(ca, 3, 2)
    ca.eval()
    _cerca(ca(lote), torch.stack((_ESPERADO_4, _ESPERADO_4)), "En modo eval el dropout no debe aplicarse")
    ca.train()
    torch.manual_seed(1)
    assert not torch.allclose(ca(lote), ca(lote)), "En modo entrenamiento debe aplicarse dropout a los pesos de atención"


def prueba_5(MultiHeadAttentionWrapper):
    torch.manual_seed(123)
    lote = torch.stack((ENTRADAS, ENTRADAS), dim=0)
    mha = MultiHeadAttentionWrapper(3, 2, context_length=6, dropout=0.0, num_heads=3)
    assert isinstance(mha.heads, nn.ModuleList) and len(mha.heads) == 3, \
        "self.heads debe ser un nn.ModuleList con num_heads cabezas"
    n = sum(p.numel() for p in mha.parameters())
    assert n == 3 * 3 * (3 * 2), f"Se esperaban {3 * 3 * 3 * 2} parámetros (3 cabezas x 3 matrices de 3x2) pero hay {n}"

    salida = mha(lote)
    assert isinstance(salida, torch.Tensor) and tuple(salida.shape) == (2, 6, 6), \
        f"Con 3 cabezas y d_out=2 la salida debe tener forma (2, 6, 6) pero tiene {getattr(salida, 'shape', None)}"
    for h, cabeza in enumerate(mha.heads):
        assert torch.allclose(salida[..., 2 * h:2 * h + 2], cabeza(lote), atol=1e-6), \
            f"Las columnas {2 * h}:{2 * h + 2} deben ser la salida de la cabeza {h} (concatena en la última dimensión)"

    _fijar_qkv(mha.heads[0], 3, 2)
    _cerca(mha(lote)[..., :2], torch.stack((_ESPERADO_4, _ESPERADO_4)), "Cada cabeza debe ser atención causal")


def prueba_6(MultiHeadAttention):
    lote = torch.stack((ENTRADAS, ENTRADAS), dim=0)
    mha = MultiHeadAttention(3, 4, context_length=6, dropout=0.0, num_heads=2)
    assert "mask" in dict(mha.named_buffers()), "La máscara debe registrarse con self.register_buffer('mask', ...)"
    _fijar_qkv(mha, 3, 4)
    _fijar(mha.out_proj, "out_proj", 4, 4, 4, sesgo=_SESGO_SALIDA)
    _cerca(mha(lote), torch.stack((_ESPERADO_6, _ESPERADO_6)), "Salida de MultiHeadAttention")

    mha = MultiHeadAttention(3, 4, context_length=12, dropout=0.0, num_heads=2)
    _fijar_qkv(mha, 3, 4)
    _fijar(mha.out_proj, "out_proj", 4, 4, 4, sesgo=_SESGO_SALIDA)
    _cerca(mha(lote), torch.stack((_ESPERADO_6, _ESPERADO_6)),
           "Con context_length=12 y entradas de 6 tokens")

    try:
        MultiHeadAttention(3, 5, context_length=6, dropout=0.0, num_heads=2)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("d_out debe ser divisible entre num_heads: lanza un error si no lo es")

    mha = MultiHeadAttention(768, 768, context_length=1024, dropout=0.0, num_heads=12)
    n = sum(p.numel() for p in mha.parameters())
    esperado = 3 * 768 * 768 + 768 * 768 + 768
    assert n == esperado, f"Con la configuración de GPT-2 small se esperaban {esperado:,} parámetros pero hay {n:,}"
    x = torch.randn(2, 10, 768)
    assert tuple(mha(x).shape) == (2, 10, 768), "La salida debe tener forma (batch, num_tokens, d_out)"
