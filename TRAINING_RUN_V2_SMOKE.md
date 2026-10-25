# Training Run V2 Smoke

Tanggal: 26 Oktober 2026

## Konfigurasi

- Base model: `izrai4/SandTube_Chat`, revision `7f21cab7f469cf15eda90548c795514d31ce87ae`
- Adapter: LoRA rank 8, target `q_proj,k_proj,v_proj,o_proj`
- Data: 340 train, 40 validasi
- Max steps: 5
- Max length: 256
- Hardware: CPU
- Output: `output/SandTube-Ai-v2-smoke`

## Hasil training

- Train loss: `2.0338809967`
- Validation loss: `1.9123432636`
- Runtime: `173.86` detik
- Adapter size: `9.034.304` byte

## Hasil benchmark

Benchmark: `evaluation_sandtube_ai_v2.json`

- Kasus: 15
- Lulus: 3
- Akurasi smoke: 20%

Per kategori:

| Kategori | Lulus | Total |
|---|---:|---:|
| Honesty | 0 | 5 |
| Missing context | 0 | 2 |
| Math | 1 | 2 |
| Identity | 2 | 2 |
| Clarification | 0 | 2 |
| Knowledge | 0 | 2 |

## Interpretasi

Smoke training membuktikan pipeline dapat berjalan dan adapter dapat disimpan, tetapi hasilnya **belum kandidat rilis**. Beberapa keluaran masih berbahasa Inggris, mengarang DOI atau isi gambar, salah menjawab pertanyaan personal, dan tidak meminta klarifikasi dengan benar. Lima langkah belum cukup untuk menilai kemampuan training penuh.

Training produksi sebaiknya dijalankan pada GPU dengan 1.000–2.000 langkah setelah data honesty, missing-context, pronomina, dan pengetahuan dasar diperkuat. Jangan memakai adapter smoke ini untuk keputusan faktual penting.

File mentah hasil evaluasi lokal disimpan sebagai `evaluation_v2_smoke.json` dan sengaja tidak harus menjadi artefak deployable.
