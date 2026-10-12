"""Pruebas de la práctica 02_01: preprocesamiento de texto para un LLM.

Cada función recibe el código del estudiante. El notebook sólo la llama.
"""

import tiktoken
import torch
from torch import nn
from torch.utils.data import DataLoader

from cllm.datos import cargar_en_paz


def prueba_1(tokenizar):
    r = tokenizar("Hello, world. Is this-- a test?")
    esperado = ["Hello", ",", "world", ".", "Is", "this", "--", "a", "test", "?"]
    assert r == esperado, f"Se esperaba {esperado} pero se obtuvo {r}"

    r = tokenizar('"It\'s the last he painted; you know," Mrs. Gisburn said.')
    esperado = ['"', "It", "'", "s", "the", "last", "he", "painted", ";", "you", "know",
                ",", '"', "Mrs", ".", "Gisburn", "said", "."]
    assert r == esperado, f"Se esperaba {esperado} pero se obtuvo {r}"

    r = tokenizar("  Mayúsculas   y\tminúsculas\n")
    assert r == ["Mayúsculas", "y", "minúsculas"], "No debes cambiar mayúsculas y debes quitar todo espacio en blanco"

    r = tokenizar("¡Vida, nada me debes! ¿Estamos en paz?")
    esperado = ["¡", "Vida", ",", "nada", "me", "debes", "!", "¿", "Estamos", "en", "paz", "?"]
    assert r == esperado, f"¡ y ¿ deben quedar como tokens aparte. Se obtuvo {r}"


def prueba_2(construir_vocabulario, preprocessed):
    v = construir_vocabulario(["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"])
    assert isinstance(v, dict), f"Se esperaba un diccionario pero se obtuvo {type(v)}"
    esperado = {"brown": 0, "dog": 1, "fox": 2, "jumps": 3, "lazy": 4, "over": 5,
                "quick": 6, "the": 7, "<|endoftext|>": 8, "<|unk|>": 9}
    assert v == esperado, f"Se esperaba {esperado} pero se obtuvo {v}"

    v = construir_vocabulario(preprocessed)
    assert len(v) == len(set(preprocessed)) + 2, "El tamaño del vocabulario no es correcto"
    assert v["<|endoftext|>"] == len(v) - 2 and v["<|unk|>"] == len(v) - 1, \
        "Los tokens especiales deben ir al final"
    assert sorted(v.values()) == list(range(len(v))), "Los IDs deben ser enteros consecutivos desde 0"


def prueba_3(SimpleTokenizerV2, vocab):
    vocab_mini = {",": 0, ".": 1, "brown": 2, "dog": 3, "the": 4, "<|endoftext|>": 5, "<|unk|>": 6}
    tok = SimpleTokenizerV2(vocab_mini)

    r = tok.encode("the brown dog.")
    assert r == [4, 2, 3, 1], f"encode('the brown dog.') debería dar [4, 2, 3, 1] pero dio {r}"
    r = tok.encode("the cat, dog")
    assert r == [4, 6, 0, 3], f"Las palabras desconocidas deben convertirse en <|unk|>. Se obtuvo {r}"
    r = tok.encode("the dog. <|endoftext|> the brown")
    assert r == [4, 3, 1, 5, 4, 2], f"<|endoftext|> no se codificó correctamente. Se obtuvo {r}"

    r = tok.decode([4, 2, 3, 1])
    assert r == "the brown dog.", f"decode([4, 2, 3, 1]) debería dar 'the brown dog.' pero dio {r!r}"
    r = tok.decode([4, 6, 0, 3])
    assert r == "the <|unk|>, dog", f"Se esperaba 'the <|unk|>, dog' pero se obtuvo {r!r}"

    vocab_es = {"¡": 0, "Vida": 1, ",": 2, "¿": 3, "paz": 4, "?": 5, "!": 6, "<|unk|>": 7}
    tok = SimpleTokenizerV2(vocab_es)
    r = tok.decode([0, 1, 2])
    assert r == "¡Vida,", f"decode debe quitar el espacio después de ¡. Se obtuvo {r!r}"
    r = tok.decode([3, 4, 5])
    assert r == "¿paz?", f"decode debe quitar el espacio después de ¿ y el de antes de ?. Se obtuvo {r!r}"

    tok = SimpleTokenizerV2(vocab)
    for texto in ("el sol acarició mi faz.", "¡Vida, nada me debes!"):
        assert tok.decode(tok.encode(texto)) == texto, f"decode(encode({texto!r})) debería recuperar el texto original"


def prueba_4(GPTDatasetV1, texto=None):
    if texto is None:
        texto = cargar_en_paz()
    tok = tiktoken.get_encoding("gpt2")
    ids = tok.encode(texto, allowed_special={"<|endoftext|>"})

    ds = GPTDatasetV1(texto, tok, max_length=4, stride=1)
    n = len(range(0, len(ids) - 4, 1))
    assert len(ds) == n, f"Con max_length=4 y stride=1 se esperaban {n} ejemplos pero hay {len(ds)}"

    ejemplo = ds[0]
    assert isinstance(ejemplo, (tuple, list)) and len(ejemplo) == 2, "__getitem__ debe regresar (entrada, objetivo)"
    x, y = ejemplo
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), "La entrada y el objetivo deben ser tensores"
    assert x.tolist() == ids[0:4], f"La primera entrada debería ser {ids[0:4]} pero es {x.tolist()}"
    assert y.tolist() == ids[1:5], f"El primer objetivo debería ser {ids[1:5]} pero es {y.tolist()}"
    assert ds[1][0].tolist() == ids[1:5], "Con stride=1 la segunda ventana debe empezar en el token 1"

    ds = GPTDatasetV1(texto, tok, max_length=8, stride=8)
    n = len(range(0, len(ids) - 8, 8))
    assert len(ds) == n, f"Con max_length=8 y stride=8 se esperaban {n} ejemplos pero hay {len(ds)}"
    x, y = ds[3]
    assert x.tolist() == ids[24:32] and y.tolist() == ids[25:33], "La ventana 3 con stride=8 no es correcta"
    x, y = ds[len(ds) - 1]
    assert len(x) == 8 and len(y) == 8, "Todos los ejemplos, incluido el último, deben tener longitud max_length"
    assert torch.equal(x[1:], y[:-1]), "El objetivo debe ser la entrada desplazada una posición"


def prueba_5(create_dataloader_v1, texto=None):
    if texto is None:
        texto = cargar_en_paz()
    ids = tiktoken.get_encoding("gpt2").encode(texto, allowed_special={"<|endoftext|>"})

    dl = create_dataloader_v1(texto, batch_size=8, max_length=4, stride=4, shuffle=False)
    assert isinstance(dl, DataLoader), f"Se esperaba un DataLoader pero se obtuvo {type(dl)}"
    assert isinstance(dl.sampler, torch.utils.data.SequentialSampler), "Con shuffle=False los datos no deben mezclarse"

    inputs, targets = next(iter(dl))
    assert inputs.shape == (8, 4) and targets.shape == (8, 4), \
        f"Se esperaban lotes de tamaño (8, 4) pero se obtuvo {tuple(inputs.shape)}"
    assert inputs.dtype == torch.long, "Los IDs deben ser enteros (torch.long)"
    assert inputs[0].tolist() == ids[0:4] and inputs[1].tolist() == ids[4:8], \
        "Las primeras filas del lote no coinciden con el texto tokenizado"
    assert torch.equal(inputs[:, 1:], targets[:, :-1]), "Los objetivos deben ser las entradas desplazadas"

    n = len(range(0, len(ids) - 4, 4))
    assert len(dl) == n // 8, "Con drop_last=True se debe descartar el último lote incompleto"

    dl = create_dataloader_v1(texto, batch_size=2, max_length=16, stride=16)
    assert isinstance(dl.sampler, torch.utils.data.RandomSampler), "Con shuffle=True los datos deben mezclarse"
    inputs, _ = next(iter(dl))
    assert inputs.shape == (2, 16), "Los parámetros batch_size y max_length no se respetaron"


def prueba_6(GPTInputEmbedding):
    torch.manual_seed(123)
    m = GPTInputEmbedding(vocab_size=50257, emb_dim=256, context_length=4)
    assert isinstance(m.tok_emb, nn.Embedding) and tuple(m.tok_emb.weight.shape) == (50257, 256), \
        "tok_emb debe ser un nn.Embedding de forma (vocab_size, emb_dim)"
    assert isinstance(m.pos_emb, nn.Embedding) and tuple(m.pos_emb.weight.shape) == (4, 256), \
        "pos_emb debe ser un nn.Embedding de forma (context_length, emb_dim)"
    assert sum(p.numel() for p in m.parameters()) == 50257 * 256 + 4 * 256, \
        "El número de parámetros entrenables no es correcto"

    entradas = torch.randint(0, 50257, (8, 4))
    out = m(entradas)
    assert isinstance(out, torch.Tensor) and tuple(out.shape) == (8, 4, 256), \
        f"Se esperaba un tensor de forma (8, 4, 256) pero se obtuvo {getattr(out, 'shape', type(out))}"
    esperado = m.tok_emb.weight[entradas] + m.pos_emb.weight[torch.arange(4, device=entradas.device)]
    assert torch.allclose(out, esperado), "La salida debe ser la suma de los embeddings de tokens y posicionales"

    m = GPTInputEmbedding(vocab_size=100, emb_dim=16, context_length=32)
    entradas = torch.tensor([[5, 5, 7, 1, 2, 3, 4, 8, 9, 10]])
    out = m(entradas)
    assert tuple(out.shape) == (1, 10, 16), "Debe funcionar con secuencias más cortas que context_length"
    assert not torch.allclose(out[0, 0], out[0, 1]), \
        "El mismo token en posiciones distintas debe tener embeddings de entrada distintos"
