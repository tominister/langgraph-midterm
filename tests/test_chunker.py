from backend.chunker import chunk_text


def test_chunker_small():
    t = "A" * 2500
    chunks = chunk_text(t, chunk_size=1000, overlap=200)
    assert len(chunks) >= 2
    # ensure no overlaps larger than expected
    for i in range(1, len(chunks)):
        assert chunks[i]['start'] < chunks[i]['end']
