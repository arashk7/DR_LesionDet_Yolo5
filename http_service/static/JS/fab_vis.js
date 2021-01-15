
window.onload = function () {
        var canvas = new fabric.Canvas('c');
        // canvas.add(new fabric.Circle({radius: 30, fill: '#f55', top: 100, left: 100}));
        // canvas.add(new fabric.Circle({radius: 30, fill: '#f55', top: 300, left: 100}));
        // var img = new fabric.Image();
        // img.setAttribute("src", 'http://www.cdgdc.edu.cn/images/zzlkpt_left1pic20180419.jpg');
        // '../static/media/12_r1.jpg'
        console.log("../../{{ filename }}")
        fabric.Image.fromURL("../../{{ filename }}", (img) => {

            img.set({
                left: 0,
                top: 0
                // Scale image to fit width / height ?
            });

            img.scaleToHeight(canvas.height);
            img.scaleToWidth(canvas.width);

            canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));

        })
        canvas.selectionColor = 'rgba(100,100,100,0.3)';
        canvas.selectionBorderColor = 'green';
        canvas.selectionLineWidth = 5;
    }