# Dinamik Atık Toplama ve Rota Optimizasyonu

Çöp konteynerlerinin doluluk oranlarına göre günlük toplama rotası hesaplayan bir
**simülatör** ve üzerine kurulu bir **prototip karar destek aracı**.

Amaç, ABC (Artificial Bee Colony) algoritmasının mevcut sabit rotaya kıyasla sağladığı
kazancı **dürüstçe ölçmektir** - kazandırmak değil, ölçmek.

Çalışma, Çanakkale'nin Gelibolu ilçe merkezinde yürütülmüştür: yol ağı ve bina
verisi bu bölgenin gerçek OpenStreetMap verisidir, talep de o binalardan türetilir.

> Simülatör bir araştırma aracı, üzerindeki uygulama bir **prototiptir**;
> sahaya inen üretim sistemi değildir.

## Veri ne kadar gerçek?

Bu ayrımın açık olması önemli, çünkü sonuçların ne kadar taşıdığını belirler.

**Gerçek olan:** yol ağı ve bina geometrisi (OpenStreetMap), kişi başı atık üretimi
ve bölge nüfusu (TÜİK), araç kapasitesi ve filo büyüklüğü.

**Gerçeğe yakın olan:** konteyner yoğunluğu. Toplama noktalarının kaç bin
taşıyacağı, belediyenin bildirdiği konteyner/nüfus oranına çapalanmıştır; model
bin başına ~44 kişi üretir, sahadaki oran ~38 kişidir (%16 sapma, aynı mertebe).

**Sentetik olan:** konteynerlerin haritadaki **konumları**. Belediyenin gerçek
konteyner koordinatları elimizde olmadığından, toplama noktaları bina
yoğunluğundan türetilir: bölge 55 m'lik bir ızgaraya bölünür, bina bulunan her
hücre bir toplama noktası olur ve nokta, o hücredeki binaların talep-ağırlıklı
merkezine yerleştirilir. Yani konumlar rastgele değildir, ama gerçek konteyner
konumları da değildir. Çalışma alanı 420 m yarıçapında bir kesittir: 187 toplama
noktası, toplam 322 bin, ~14.100 sakin.

**Varsayım olan:** yakıt katsayıları. Belediyeden gerçek yakıt tüketimi
alınamadığı için literatürden türetilmiştir. Bu nedenle mutlak litre/CO₂
rakamları tahmindir; çözücüler arası **göreli** karşılaştırma ise tüm çözücüler
aynı katsayıları paylaştığı için bu seçime büyük ölçüde dayanıklıdır.

## Sonuç özeti

5 çözücü eşit duvar saatiyle, 90 gün × 10 tekrar karşılaştırıldı.
Manşet metrik **yakıt/CO₂**'dir.

| Kod | Çözücü | Yakıt (L/gün) | B0'a karşı | Durak |
|---|---|---|---|---|
| B0 | Sabit rota (mevcut durum) | 35,13 | - | 187,0 |
| B1 | Eşik + greedy | 33,19 | −%5,5 | 143,2 |
| **B2** | **OR-Tools** | **32,04** | **−%8,8** | 143,2 |
| X1 | ABC (temel) | 33,70 | −%4,1 | 144,5 |
| X2 | ABC + yerel arama | 33,87 | −%3,6 | 148,3 |

Üç bulgu, olumsuz oldukları için gizlenmedi, sayısallaştırıldı:

- **ABC'nin teorik avantaj alanı yok.** OR-Tools'un çözemediği yük–mesafe bağlaşımı
  toplam yakıtın ~%1'i ve çözücüler arasında farklılaşmıyor. Yük momentinin %79'u
  döküm bacağında ve sıralamayla değiştirilemez.
- **Atlama cezası (λ) etkisiz.** Doluluk-farkındalıklı zorunlu ziyaret kuralı kararı
  zaten belirliyor; çözücü atlayabildiği her konteyneri zaten atlıyor.
- **En eyleme dönüştürülebilir bulgu algoritmik değil.** Mesafenin %84'ü garaj–bölge–döküm
  gidiş-gelişi. Asıl kaldıraç rotalama değil, **tesis konumlandırması**.

## Kurulum

```bash
uv sync
```

## Çalıştırma

İki ayrı arayüz vardır, iki farklı amaca hizmet ederler:

```bash
uv run uvicorn api.main:app --port 8000    # KARAR DESTEK ARACI  -> localhost:8000
uv run streamlit run ui/dashboard.py       # ARAŞTIRMA PANELİ    -> localhost:8501
```

**Karar destek aracı** (FastAPI + Leaflet) günlük operasyon içindir:
doluluk gir → bugünü çöz → planı incele → uygula ve kaydet. Durum günden güne taşınır.

**Araştırma paneli** (Streamlit) deney sonuçlarını gösterir: KPI'lar, karşılaştırma
haritaları, sağlık raporu.

Deneyi yeniden üretmek için:

```bash
uv run python run_full.py    # tam deney -> runs/<zaman>_<confighash>/
```

`runs/` klasörü versiyonlanmaz; yukarıdaki komut onu üretir. Araştırma paneli
sonuçları oradan okur, bu yüzden temiz bir klonda önce deneyi koşmak gerekir.
Karar destek aracı `runs/` olmadan da çalışır (varsayılan λ = 0,1).

## Geliştirme

```bash
uv run pytest                 # 125 test
uv run ruff check .           # lint
uv run mypy domain solvers    # tip denetimi (strict)
```

## Mimari

```
domain/     problem.py, solution.py, evaluator.py   <- ÇEKİRDEK (çıplak diziler)
solvers/    greedy, threshold_greedy, ortools, abc
sim/        engine, containers, experiment, operations
api/ web/   karar destek aracı (FastAPI + Jinja2 + Leaflet)
ui/         araştırma paneli (Streamlit + Folium)
data/       OSM yükleme, mesafe matrisi, cache
```

Kod tabanı dört kurala göre yazılmıştır; hepsi testlerle korunmaktadır.

**1. Çözücüler değiştirilebilir.** Hepsi `solve(problem: VRPProblem) -> Solution`
arayüzünü uygular. Simülatör hangisinin çalıştığını bilmez; bir çözücüyü
değiştirmek sistemin geri kalanını etkilemez.

**2. Evaluator tek doğruluk kaynağıdır.** Hiçbir çözücü, arayüz ya da API kendi
maliyetini hesaplamaz. Maliyet formülü ve fizibilite kontrolü yalnızca
`domain/evaluator.py` içinde yaşar; OR-Tools'un döndürdüğü rota bile oradan geçer.
İki test bunu kilitler: OR-Tools'un raporladığı amaç değeri ile Evaluator'ın
hesabı tam sayı olarak eşit olmalıdır, ve API'nin döndürdüğü yakıt doğrudan
`Simulator` + `Evaluator` sonucuyla birebir aynı olmalıdır.

**3. Çekirdek çıplak dizilerle çalışır.** `domain/` ve ABC'nin iç döngüsü yalnızca
`np.ndarray` ve skaler sayı kullanır; sınıf, sözlük, `None` yoktur. Nesne dünyası
ile dizi dünyası arasındaki geçiş günde bir kez, yalnızca iki fonksiyonda olur:
`build_problem()` ve `decode_solution()`. Rota değerlendirilirken dönüşüm yapılmaz.

**4. Birimler istisnasız tam sayıdır.** Mesafe metre, süre saniye, hacim litre,
yakıt mililitre. Gerekçe teknik: OR-Tools ondalık kabul etmez ve ondalık sızarsa
hata vermeden sessizce yanlış sonuç üretir.

Günlük plan iş akışı **tek bir yerde** bulunur. Zorunlu ziyaret kararı ve gün
çözümü `Simulator.solve_day` içindedir; 90 günlük simülasyon döngüsü de aynı
fonksiyonu çağırır. Çözücü seçimi tek bir fabrikadan gelir. Böylece araç ile
simülasyon aynı davranışı paylaşır.

---
*Veri kaynakları: © OpenStreetMap katkıda bulunanlar (ODbL); TÜİK ADNKS 2025;
TÜİK Belediye Atık İstatistikleri
