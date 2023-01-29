console.log("loaded script successfully")

dirnames = [
    'k1_0.35_k2_0.91_k3_0.06_step0',
    'k1_0.95_k2_0.11_k3_0.12_step850',
    'k1_0.96_k2_0.11_k3_0.12_step1000',
    'k1_0.97_k2_0.11_k3_0.11_step950',
    'k1_0.97_k2_0.11_k3_0.12_step900',
    'k1_0.97_k2_0.86_k3_0.09_step50',
    'k1_0.98_k2_0.11_k3_0.12_step650',
    'k1_0.98_k2_0.11_k3_0.12_step700',
    'k1_0.98_k2_0.11_k3_0.12_step750',
    'k1_0.98_k2_0.11_k3_0.12_step800',
    'k1_0.99_k2_0.11_k3_0.13_step600',
    'k1_1.00_k2_0.12_k3_0.14_step500',
    'k1_1.00_k2_0.12_k3_0.14_step550',
    'k1_1.00_k2_0.14_k3_0.15_step450',
    'k1_1.00_k2_0.19_k3_0.16_step400',
    'k1_1.00_k2_0.31_k3_0.17_step350',
    'k1_1.00_k2_0.43_k3_0.17_step300',
    'k1_1.00_k2_0.54_k3_0.17_step250',
    'k1_1.00_k2_0.64_k3_0.16_step200',
    'k1_1.00_k2_0.72_k3_0.15_step150',
    'k1_1.00_k2_0.80_k3_0.12_step100'
]

anchorEl = document.querySelectorAll("#anchor")[0]
pEl = document.querySelectorAll("#label")[0]
console.log(anchorEl)
img_count = 0
function updateImage() {
    anchorEl.src = "../im/" + dirnames[0] + `/${img_count}.png`
    pEl.innerHTML = `Step: ${img_count}`
    img_count = img_count + 100
    if (img_count === 50_000) {
        clearInterval(update_scheduler)
    }
}

update_scheduler = setInterval(updateImage, 100)