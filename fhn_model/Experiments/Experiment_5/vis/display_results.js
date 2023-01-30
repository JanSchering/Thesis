console.log("loaded script successfully")

var dirnames = [
    'k1_0.35_k2_0.91_k3_0.06_step0',
    'k1_0.97_k2_0.86_k3_0.09_step50',
    'k1_1.00_k2_0.80_k3_0.12_step100',
    'k1_1.00_k2_0.72_k3_0.15_step150',
    'k1_1.00_k2_0.64_k3_0.16_step200',
    'k1_1.00_k2_0.54_k3_0.17_step250',
    'k1_1.00_k2_0.43_k3_0.17_step300',
    'k1_1.00_k2_0.31_k3_0.17_step350',
    'k1_1.00_k2_0.19_k3_0.16_step400',
    'k1_1.00_k2_0.14_k3_0.15_step450',
    'k1_1.00_k2_0.12_k3_0.14_step500',
    'k1_1.00_k2_0.12_k3_0.14_step550',
    'k1_0.99_k2_0.11_k3_0.13_step600',
    'k1_0.98_k2_0.11_k3_0.12_step650',
    'k1_0.98_k2_0.11_k3_0.12_step700',
    'k1_0.98_k2_0.11_k3_0.12_step750',
    'k1_0.98_k2_0.11_k3_0.12_step800',
    'k1_0.95_k2_0.11_k3_0.12_step850',
    'k1_0.97_k2_0.11_k3_0.12_step900',
    'k1_0.97_k2_0.11_k3_0.11_step950',
    'k1_0.96_k2_0.11_k3_0.12_step1000',
]

var img_count = 0
var update_scheduler;

function createResultContainer(dirname) {
    var metaInfo = dirname.split("_")
    var imEl = document.createElement("img")
    imEl.src = "../im/" + dirname + `/${img_count}.png`
    imEl.id = metaInfo[6] 
    imEl.classList.add("vis")
    var pEl = document.createElement("p")
    pEl.innerHTML = `Gradient Descent Step: ${metaInfo[6].replace("step", "")}`
    pEl.classList.add("label")
    var containerEl = document.createElement("div")
    containerEl.classList.add("subContainer")
    containerEl.appendChild(pEl)
    containerEl.appendChild(imEl)

    anchorEl = document.querySelectorAll("#anchor")[0]
    anchorEl.appendChild(containerEl)
}

dirnames.forEach(dirname => {
    createResultContainer(dirname)
})


var anchorEl = document.querySelectorAll("#anchor")[0]

var stepCounter = document.createElement("div")
var stepCounterInner = document.createElement("p")
stepCounterInner.innerHTML = `Simulation Step: ${img_count}`
stepCounterInner.classList.add("stepCounterInner")
stepCounter.appendChild(stepCounterInner)
stepCounter.classList.add("stepCounter")
var subContainerStep = document.createElement("div")
subContainerStep.classList.add("subContainer")
subContainerStep.appendChild(stepCounter)
anchorEl.appendChild(subContainerStep)

function updateImages() {
    imContainers = []
    dirnames.forEach(dirname => {
        var metaInfo = dirname.split("_")
        var id = metaInfo[6]
        imContainers.push(document.querySelectorAll(`#${id}`)[0])
    })
    dirnames.forEach((dirname, idx) => {
        imContainers[idx].src = "../im/" + dirname + `/${img_count}.png`
    })
    img_count = img_count + 100
    stepCounter.innerHTML = `Simulation Step: ${img_count}`
    if (img_count === 50_000) {
        clearInterval(update_scheduler)
    }
}

var startButton = document.createElement("button")
startButton.innerHTML = "start"
startButton.addEventListener("click", () => {
    update_scheduler = setInterval(updateImages, 100)
})
startButton.classList.add("playButton")
var subContainer = document.createElement("div")
subContainer.classList.add("subContainer")
subContainer.appendChild(startButton)
anchorEl.appendChild(subContainer)



