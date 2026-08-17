# Dinamik Atık Toplama ve Rota Optimizasyonu

Çöp konteynerlerinin doluluk oranlarına göre günlük toplama rotası hesaplayan bir
**simülatör** ve üzerine kurulu bir **prototip karar destek aracı**.

Amaç, doluluk-tabanlı akıllı toplamanın mevcut sabit rotaya kıyasla sağladığı
kazancı **dürüstçe ölçmektir** - kazandırmak değil, ölçmek.

Karşılaştırılan yöntemler: bir eşik kuralı, iki metasezgisel (Yapay Arı Kolonisi
/ ABC) ve referans üst sınır olarak Google OR-Tools. Hiçbiri makine öğrenmesi
değildir; hepsi arama ve sezgisel optimizasyon ailesindendir - eğitim verisi,
model ya da öğrenilen ağırlık yoktur.

Çalışma, Çanakkale'nin Gelibolu ilçe merkezinde yürütülmüştür: yol ağı ve bina
verisi bu bölgenin gerçek OpenStreetMap verisidir, talep de o binalardan türetilir.

> Simülatör bir araştırma aracı, üzerindeki uygulama bir **prototiptir**;
> sahaya inen üretim sistemi değildir.

## Veri ne kadar gerçek?

Bu ayrımın açık olması önemli, çünkü sonuçların ne kadar taşıdığını belirler.

**Gerçek olan:** yol ağı ve bina geometrisi (OpenStreetMap), kişi başı atık üretimi
ve bölge nüfusu (TÜİK).

**Gerçeğe yakın olan:** ölçek. Çalışma alanı, OSM'in kentsel doku poligonuyla
kırpılır (5,07 km²) - yapay bir yarıçap değil, merkezin kendisi. Model 32.592
sakin üretir; TÜİK merkez nüfusu 33.062 (%99). Filo ihtiyacı ortalama 5,1 /
tepe 7,0 araç çıkar ve bu, bu büyüklükte bir merkez için beklenen mertebeyle
tutarlıdır.

**Sentetik olan:** konteynerlerin haritadaki **konumları**. Gerçek konteyner
koordinatları açık veri olarak bulunmadığından toplama noktaları bina
yoğunluğundan türetilir: bölge 55 m'lik ızgaraya bölünür, bina bulunan her hücre
bir nokta olur ve nokta o hücrenin talep-ağırlıklı merkezine yerleşir. Yani
konumlar rastgele değildir, ama gerçek konteyner konumları da değildir.
Ölçek: **784 toplama noktası, 1.019 bin, 7 araç**.

**Varsayım olan:** yakıt katsayıları ve referans operasyon ölçeği (konteyner
sayısı, filo). Yakıt katsayıları literatürdeki saha ölçümlerinden türetilmiştir;
mutlak litre/CO₂ rakamları tahmindir, çözücüler arası **göreli** karşılaştırma
ise tüm çözücüler aynı katsayıları paylaştığı için bu seçime büyük ölçüde
dayanıklıdır. Referans ölçek değerleri de bağımsız bir sayımla doğrulanmamıştır;
modelin girdisi değil, sınandığı paydadır.

## Çözücüler nasıl çalışıyor?

Beşi de aynı arayüzü uygular (`solve(problem) -> solution`) ve maliyetleri aynı
bağımsız değerlendiriciden geçer; simülatör hangisinin koştuğunu bilmez.

| Kod | Yöntem | Nasıl karar veriyor |
|---|---|---|
| `B0` | Sabit rota | Her gün **tüm** konteynerlere gider. Rota en yakın komşu ile kurulur. Mevcut durumun temsili. |
| `B1` | Eşik + greedy | Tek kural: doluluk %70'in üstündeyse topla. Arama yapmaz, anında sonuç verir. |
| `B2` | Google OR-Tools | Kısıt programlama + **Guided Local Search**. Referans üst sınır. |
| `X1` | Yapay Arı Kolonisi (ABC) | Popülasyon tabanlı metasezgisel; projenin odak algoritması. |
| `X2` | ABC + yerel arama | X1 üzerine kısa 2-opt. Ablasyon varyantı. |

**OR-Tools (B2)** üç adımda çalışır. Önce problem bir kısıt modeline çevrilir:
düğümler, ark maliyetleri ve üzerlerinde üst sınır olan "boyutlar" (kapasite,
vardiya süresi); atlanabilir konteynerler *disjunction* olarak tanımlanır - "bu
düğümü ziyaret et ya da şu cezayı öde". Ardından açgözlü bir sezgisel ilk fizibil
çözümü kurar (konteyneri maliyeti en az artıracak yere yerleştirerek). Asıl iş
son adımdadır: **Guided Local Search** rotaya yerel hamleler uygular (2-opt,
or-opt, relocate, swap) ve yerel optimuma takıldığında o çözümdeki pahalı
özellikleri *cezalandırarak* amaç fonksiyonunu geçici olarak değiştirir, böylece
aramayı o çukurdan çıkmaya zorlar. Duvar saati dolunca en iyi çözüm döner.

**ABC (X1/X2)** ile farkı arama stratejisidir: ABC bir **popülasyon** tutar ve
arı metaforuyla (işçi / gözcü / kâşif) çözüm uzayını tarar; GLS **tek** bir
çözüm tutar ve ceza biriktirerek yön değiştirir. İkisi de aynı yerel hamle
ailesini kullanır.

> **Hiçbiri makine öğrenmesi değildir.** Eğitim verisi, model ya da öğrenilen
> ağırlık yoktur; hepsi klasik arama ve sezgisel optimizasyon yöntemleridir.
> Aynı girdi ve aynı bütçeyle aynı işi yaparlar. Aradaki asıl performans farkının
> bir kısmı da algoritmik değildir: OR-Tools C++ çekirdekte, ABC saf Python'da
> çalışır ve eşit duvar saati protokolü bu yüzden ABC'yi dezavantajlı konuma
> sokar.

## Sonuç özeti

5 çözücü, 10 tekrar × 90 gün. Manşet metrik **yakıt/CO₂**'dir.

Deney iki bütçede koşuldu ve **ikisi farklı sonuç verdi** - bu farkın kendisi
bulgudur:

| Çözücü | 60 sn bütçe | 30 sn bütçe |
|---|---|---|
| Sabit rota (mevcut durum) | - | - |
| **OR-Tools** | **−%24,4** | −%11,0 |
| **Eşik + greedy** | −%18,5 | **−%18,7** |
| ABC (temel) | −%14,3 | −%13,6 |
| ABC + yerel arama | −%10,7 | −%9,1 |

**OR-Tools süreye çok duyarlıdır.** 784 konteynerlik problemde 60 saniyede açık
ara en iyi sonucu verir; 30 saniyede belirgin biçimde zayıflar ve 900 günün
4'ünde hiç çözüm bulamaz. **Eşik kuralı ise süreden hiç etkilenmez** (−%18,5 ve
−%18,7) çünkü tek bir kuraldır, arama yapmaz.

Pratik sonucu şudur: günlük planlamada 60 saniye beklemek sorun değilse OR-Tools,
anlık cevap gerekiyorsa eşik kuralı. Karar destek aracı bu yüzden varsayılan
olarak OR-Tools + 60 saniye ile gelir.

Diğer bulgular:

- **ABC'nin teorik avantaj alanı yok.** OR-Tools'un çözemediği yük-mesafe
  bağlaşımı toplam yakıtın %1'inin altında (%0,51–0,84) ve çözücüler arasında
  farklılaşmıyor. Yük momentinin %79'u döküm bacağında ve sıralamayla
  değiştirilemez.
- **Naif eşik kuralı, projenin odak algoritmasını (ABC) geçiyor.** Her metrikte
  ve her bütçede. Olumsuz olduğu için gizlenmedi.
- **Hijyen tavanı bir güvenlik parametresi, tasarruf kaldıracı değil.**
  3 gün → −%18,7, 5 gün → −%24,2, 7 gün → −%19,2. Beklenen monoton davranış
  yok (7 gün, 5 günden kötü), yani tavanı gevşetmek güvenilir bir kazanç
  getirmiyor. Çalışma noktası en muhafazakâr değerde (3 gün) bırakıldı.
- **Taşma sıfır değil: garanti olasılıksal.** 900 günde iki olay (B1 ve X1'de
  birer tane). Güvenlik kuralı taşmayı imkânsız kılmıyor, olasılığını `k` ile
  belirliyor. Ölçüldü: k=4 → 1 olay, k=5 → 0 (bedeli günde ~36 ek durak).
- **Deadhead toplam mesafenin %63'ü.** Optimize edilebilir pay %37; en büyük
  kaldıraç algoritma değil, döküm sahasının konumu.

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
