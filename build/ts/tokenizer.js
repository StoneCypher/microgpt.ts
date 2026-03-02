export function buildTokenizer(docs) {
    const charSet = new Set();
    for (const doc of docs) {
        for (const ch of doc) {
            charSet.add(ch);
        }
    }
    const uchars = [...charSet].sort(), BOS = uchars.length, vocabSize = uchars.length + 1;
    return { uchars, BOS, vocabSize };
}
export function tokenize(doc, uchars, BOS) {
    return [BOS, ...Array.from(doc ?? '').map(ch => uchars.indexOf(ch)), BOS];
}
