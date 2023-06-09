function sendImg(tab, imgPath) {
  const handler = () => {
    let target;

    if (tab === 'inpaint') {
      target = 'img2img_prompt_image';
    }

    const imgParent = gradioApp().getElementById(target);
    const fileInput = imgParent.querySelector('input[type="file"]');
    if (fileInput) {
      fileInput.files = imgPath;
      fileInput.dispatchEvent(new Event('change'));
    }
  };

  // Execute code after the image is saved (after the python function handler on the button)
  setTimeout(handler, 500);
}

function getImgFromInpaintTab() {
  const element = document.querySelector('#img2img_img2img_tab img');

  return element?.src;
}
