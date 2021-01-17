let randomColor;

function setBackgraundColor(){
  randomColor =  Math.floor(Math.random()*16777215).toString(16);
  document.body.style.backgroundColor = "#" + randomColor;
}

function saveOrNotColorChoosen(choosen){
  if(randomColor){
    if(choosen){
      colorChoosen.push("#" + randomColor);
      setBackgraundColor();
    }else{
      colorNotChoosen.push("#" + randomColor);
      setBackgraundColor();
      }
    }else{setBackgraundColor();}
  }
