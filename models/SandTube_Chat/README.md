# SandTube_Chat — salinan model dasar dari Hugging Face

Berkas konfigurasi, tokenizer, template chat, dan model card di folder ini berasal dari `izrai4/SandTube_Chat` pada revision `7f21cab7f469cf15eda90548c795514d31ce87ae`.

Bobot sumber Hugging Face bernama `model.safetensors` dan berukuran 2.200.119.664 byte. Agar tiap berkas memenuhi batas ukuran Git LFS GitHub Free/Pro, bobot tersebut dipecah tanpa mengubah byte menjadi dua berkas LFS:

- `model.safetensors.part-00`
- `model.safetensors.part-01`

Setelah clone repo dengan Git LFS terpasang, gabungkan kembali bobot asli:

```bash
python restore_model.py
```

Skrip membuat `model.safetensors` dan memverifikasi SHA-256 `30b53e1e9697f742d67fed7ac94efabfc664786a596f7b0fd5764a32727d1b64`. Rincian hash tiap shard ada di `SHA256SUMS.json`. `MODEL_CARD.md` adalah salinan kartu model Hugging Face.

Kartu model upstream saat ini belum menyatakan lisensi. Periksa syarat pemilik model sebelum menggunakan atau mendistribusikan ulang berkas ini.
