
import { train } from './index.mjs';

let signal = { aborted: false };

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === 'stop') {
    signal.aborted = true;
    return;
  }

  if (msg.type === 'train') {
    signal = { aborted: false };

    try {
      const finalSamples = await train({
        ...msg.config,
        signal,
        onStep(update) {
          self.postMessage({ type: 'step', ...update });
        },
      });
      self.postMessage({ type: 'done', samples: finalSamples });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }
};
