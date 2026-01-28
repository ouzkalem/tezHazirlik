---
description: Otomatik commit ve push - milestone tamamlandığında
---

# Otomatik Git Commit ve Push Kuralı

Bu workflow, bir milestone tamamlandığında otomatik olarak çalıştırılmalıdır.

## Adımlar

// turbo-all

1. Değişiklikleri stage et:
```bash
git add -A
```

2. Commit yap (milestone açıklaması ile):
```bash
git commit -m "[MILESTONE] <açıklama>"
```

3. Push et:
```bash
git push origin main
```

## Notlar

- Her notebook tamamlandığında commit yapılmalı
- Önemli değişikliklerden sonra push edilmeli
- Commit mesajı Türkçe ve açıklayıcı olmalı
