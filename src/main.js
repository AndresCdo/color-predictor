const colors = [];

function setBackgroundColor() {
  const randomColor = Math.floor(Math.random() * 16777215).toString(16);
  document.body.style.backgroundColor = "#" + randomColor;
}

function saveOrNotColorChoosen(choosen) {
  colors.push({ color: document.body.style.backgroundColor, choosen });
  setBackgroundColor();
}