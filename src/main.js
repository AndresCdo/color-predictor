const colors = [];

function setBackgroundColor() {
  const randomColor = getRandomColor();
  document.body.style.backgroundColor = randomColor;
}

function saveColorChoice(isChosen) {
  const currentColor = document.body.style.backgroundColor;
  colors.push({ color: currentColor, isChosen });
  setBackgroundColor();
}

function getRandomColor() {
  const randomColor = Math.floor(Math.random() * 16777215).toString(16);
  return "#" + randomColor;
}