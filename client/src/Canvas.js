import React, { useRef, useEffect, useState } from 'react';
import CanvasDraw from 'react-canvas-draw';
import styled from 'styled-components';
import axios from 'axios';

const WIDTH = 280;
const HEIGHT = 280;
const RADIUS = 12;

const Wrapper = styled.div`
  // border-radius: 10px;
  overflow: hidden;
  margin: auto;
  -webkit-box-shadow: 0px 4px 9px 4px rgba(0, 0, 0, 0.29);
  box-shadow: 0px 4px 9px 4px rgba(0, 0, 0, 0.29);
  position: relative;
  text-align: center;
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
`;

const Canvas = (props) => {
  const canvasRef = useRef();
  const [showText, setShowText] = useState(true);

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
    canvasRef.current.clear();
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

  const handleChange = (e) => {
    props.setShowResult(true);
    props.setResult('');
    props.setShuffling(true);
    let ctx = e.ctx.drawing;
    let bounds = getBounds(canvasRef.current.lines[0].points);
    let bounded = ctx.getImageData(
      bounds.minX,
      bounds.minY,
      bounds.maxX,
      bounds.maxY
    );

    canvasRef.current.clear();
    ctx.putImageData(
      bounded,
      WIDTH / 2 - (bounds.maxX - bounds.minX) / 2,
      HEIGHT / 2 - (bounds.maxY - bounds.minY) / 2
    );

    let data = ctx.getImageData(0, 0, WIDTH, HEIGHT);

    let converted = convertImageData(data);
    let json = JSON.stringify(converted);
    axios
      .post('http://10.0.0.19:3001/predict', json, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json',
        },
      })
      .then((res) => {
        setTimeout(() => {
          props.setShuffling(false);
          props.setResult(String(res.data));
        }, 400);
      })
      .catch((err) => {
        console.log(err);
      });
  };

  return (
    <Wrapper onPointerDown={handleDrawStart}>
      <CanvasDraw
        canvasWidth={WIDTH}
        canvasHeight={HEIGHT}
        ref={canvasRef}
        onChange={handleChange}
        brushRadius={RADIUS}
        brushColor={0x000000}
        lazyRadius={0}
        hideGrid
        // style={{ imageRendering: 'pixelated' }}
        hideInterface={true}
      />
      {showText && <StartText>Draw here</StartText>}
    </Wrapper>
  );
};

export default Canvas;
