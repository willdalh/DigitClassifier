import React, { useEffect, useState, useRef } from 'react';
import Canvas from './Canvas';
import styled from 'styled-components';

const Wrapper = styled.div`
  display: grid;
  width: 75%;
  margin: 0 auto;
  align-self: center;
  grid-auto-flow: row;
  align-items: center;
`;

const Title = styled.h1`
  text-align: center;
`;

const YouDrewText = styled.div`
  text-align: center;
  font-size: 60px;
`;

const ResultText = styled.div`
  text-align: center;
  font-size: 60px;
`;

const App = () => {
  const [showResult, setShowResult] = useState(false);
  const [result, setResult] = useState('');
  const [placeholder, setPlaceholder] = useState('');
  const [requesting, setRequesting] = useState(false);
  const interval = useRef(null);

  useEffect(() => {
    if (requesting) {
      interval.current = setInterval(() => {
        setPlaceholder(
          String(Number(placeholder) + Number.parseInt(Math.random() * 9)) % 10
        );
      }, 100);
    } else {
      clearInterval(interval.current);
    }
  }, [requesting]);
  return (
    <Wrapper>
      <Title>Digit Classifier</Title>

        <Canvas
          setResult={setResult}
          setShowResult={setShowResult}
          setRequesting={setRequesting}
          requesting={requesting}
        />

      {showResult && (
        <ResultText>
          <YouDrewText>You drew: </YouDrewText>
          <ResultText>{requesting ? placeholder : result}</ResultText>
        </ResultText>
      )}
    </Wrapper>
  );
};

export default App;
