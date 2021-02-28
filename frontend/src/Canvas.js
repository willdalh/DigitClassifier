import React, { useRef, useState } from 'react';
import CanvasDraw from 'react-canvas-draw';
import styled, { keyframes } from 'styled-components';
import axios from 'axios';

const WIDTH = 280;
const HEIGHT = 280;
const RADIUS = 12;

const dummyCanvas = document.createElement('canvas');
dummyCanvas.setAttribute('width', WIDTH);
dummyCanvas.setAttribute('height', HEIGHT);

const appear = keyframes`
  0% {
    transform: translateY(-100%);
    border-radius: 100px;
    color: #ffffff;
  }
  90% {
    color: #ffffff;
  }
  100% {
    color: #cccccc;
    border-radius: 0px;
    transform: translateY(0%);
  }
`;

const Wrapper = styled.div`
  display: grid;
  align-items: center;
  // overflow: hidden;
`;

const CanvasWrapper = styled.div`
  // border-radius: 100px;
  overflow: hidden;
  margin: auto;

  -webkit-box-shadow: 0px 4px 9px 4px rgba(0, 0, 0, 0.29);
  box-shadow: 0px 4px 9px 4px rgba(0, 0, 0, 0.29);
  position: relative;
  text-align: center;
  animation: 0.7s ${appear} ease;
`;

const StartText = styled.div`
  // pointer-events: none;
  top: 90px;
  font-size: 30px;
  text-align: center;
  left: 0;
  right: 0;
  align-self: center;
  position: absolute;
  z-index: 100;
  pointer-events: none;
  color: #333333;
  font-weight: bold;
  width: 70%;
  margin: 0 auto;
`;

const Canvas = (props) => {
  const canvasRef = useRef();
  const [showText, setShowText] = useState(true);

  const stillDrawing = useRef(false);
  const stillDrawingTimeout = useRef(null);
  const performRequestTimeout = useRef(null);

  const convertImageData = (data) => {
    let converted = new Array(WIDTH / 10)
      .fill(0)
      .map(() => new Array(HEIGHT / 10).fill(0));
    data = data.data;
    for (let x = 0; x < WIDTH / 10; x++) {
      for (let y = 0; y < HEIGHT / 10; y++) {
        let n = y * 10 * WIDTH * 4 + x * 10 * 4;
        if (data[n] === 255 && data[n + 1] === 255 && data[n + 2] === 255) {
          converted[(x, y)] = 255;
        } else {
          converted[x][y] = Number.parseInt(
            ((data[n + 3] / 255) *
              (3 * 255 - data[n] - data[n + 1] - data[n + 2])) /
              3
          );
        }
      }
    }
    return converted;
  };

  const handleDrawStart = () => {
    if (showText) setShowText(false);
    props.setShowResult(false);
    // console.log('In drawstart', stillDrawing.current);
    if (!stillDrawing.current) {
      canvasRef.current.clear();
    }
  };

  const handleDrawEnd = () => {
    stillDrawing.current = true;
    clearTimeout(stillDrawingTimeout.current);
    stillDrawingTimeout.current = setTimeout(() => {
      stillDrawing.current = false;
    }, 800);
  };

  const getBounds = (points) => {
    let xPoints = points.map((e) => e.x);
    let yPoints = points.map((e) => e.y);
    let minX = Math.min(...xPoints) - RADIUS;
    let maxX = Math.max(...xPoints) + RADIUS;
    let minY = Math.min(...yPoints) - RADIUS;
    let maxY = Math.max(...yPoints) + RADIUS;
    return { minX, maxX, minY, maxY };
  };

  const performRequest = (e) => {
    // console.log('Waiting for more drawing...');
    clearTimeout(performRequestTimeout.current);
    performRequestTimeout.current = setTimeout(() => {
      // console.log('In performrequest', stillDrawing.current);
      if (!stillDrawing.current) {
        // console.log('Sending now');
        props.setShowResult(true);
        props.setResult('');
        props.setRequesting(true);
        let ctx = e.ctx.drawing;
        e.canvas.interface.getContext('2d').clearRect(0, 0, WIDTH, HEIGHT);
        let bounds = getBounds(canvasRef.current.lines[0].points);
        let bounded = ctx.getImageData(
          bounds.minX,
          bounds.minY,
          bounds.maxX,
          bounds.maxY
        );
        let newCtx = dummyCanvas.getContext('2d');

        newCtx.putImageData(
          bounded,
          WIDTH / 2 - (bounds.maxX - bounds.minX) / 2,
          HEIGHT / 2 - (bounds.maxY - bounds.minY) / 2
        );

        let data = newCtx.getImageData(0, 0, WIDTH, HEIGHT);
        newCtx.clearRect(0, 0, WIDTH, HEIGHT);

        let converted = convertImageData(data);
        let json = JSON.stringify(converted);
        axios
          .post('https://willdalh.xyz/api/predict', json, {
            headers: {
              'Access-Control-Allow-Origin': '*',
              'Content-Type': 'application/json',
            },
          })
          .then((res) => {
            setTimeout(() => {
              props.setRequesting(false);
              props.setResult(String(res.data));
            }, 400);
          })
          .catch((err) => {
            console.log(err);
          });
      }
    }, 800);
  };

  return (
    <Wrapper>
      <CanvasWrapper
        onPointerDown={() => {
          if (!props.requesting) {
            handleDrawStart();
          }
        }}
        onPointerUp={() => {
          if (!props.requesting) {
            handleDrawEnd();
          }
        }}
      >
        <CanvasDraw
          canvasWidth={WIDTH}
          canvasHeight={HEIGHT}
          ref={canvasRef}
          onChange={performRequest}
          brushRadius={RADIUS}
          brushColor={'#000000'}
          catenaryColor={'#0a0302'}
          lazyRadius={0}
          hideGrid
          disabled={props.requesting}
          style={{
            backgroundColor: props.requesting ? '#dddddd' : 'white',
            transition: 'ease 0.2s',

          }}
          // style={{ imageRendering: 'pixelated' }}
          hideInterface={props.requesting}
        />
        {showText && <StartText>Draw a digit here</StartText>}
      </CanvasWrapper>
    </Wrapper>
  );
};

export default Canvas;
