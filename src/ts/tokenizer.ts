/**
 * Character-level tokenizer for MicroGPT.
 *
 * @module tokenizer
 */





/**
 * Build a character-level tokenizer from a list of documents.
 *
 * @param docs  Array of document strings.
 * 
 * @returns  `uchars` (sorted unique characters), `BOS` token id, and `vocabSize`.
 */

export function buildTokenizer(docs: string[]): {
  uchars: string[];
  BOS: number;
  vocabSize: number;
} {

  const charSet = new Set<string>();

  for (const doc of docs) {
    for (const ch of doc) {
      charSet.add(ch);
    }
  }

  const uchars    = [...charSet].sort(),
        BOS       = uchars.length,
        vocabSize = uchars.length + 1;

  return { uchars, BOS, vocabSize };

}



/**
 * Tokenize a document string into an array of token ids, surrounded by BOS.
 *
 * @param doc     The document string.
 * @param uchars  Sorted unique characters (from {@link buildTokenizer}).
 * @param BOS     The BOS token id.
 * 
 * @returns  Array of token ids: `[BOS, ...charIds, BOS]`.
 */

export function tokenize(doc: string | undefined, uchars: string[], BOS: number): number[] {
  return [BOS, ...Array.from(doc ?? '').map(ch => uchars.indexOf(ch)), BOS];
}
