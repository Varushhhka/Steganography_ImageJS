import { Image } from 'image-js';
import { readFileSync, writeFileSync } from 'fs';

function bytesToBits(bytes) {
    const bits = [];
    for (const byte of bytes) {
        for (let i = 7; i >= 0; i--) {
            bits.push((byte >> i) & 1);
        }
    }
    return bits;
}


function bitsToBytes(bits) {
    const bytes = [];
    for (let i = 0; i < bits.length; i += 8) {
        let byte = 0;
        for (let j = 0; j < 8; j++) {
            if (i + j < bits.length) {
                byte = (byte << 1) | bits[i + j];
            } else {
                byte = byte << 1;
            }
        }
        bytes.push(byte);
    }
    return new Uint8Array(bytes);
}

function readStdin() {
    return new Promise((resolve) => {
        let data = '';
        process.stdin.setEncoding('utf8');
        process.stdin.on('data', chunk => data += chunk);
        process.stdin.on('end', () => resolve(data.trim()));
    });
}

async function embedData(inputPath, outputPath, payloadHex) {
    try {
        const buffer = readFileSync(inputPath);
        const image = await Image.load(buffer);

        const payloadBytes = Uint8Array.from(payloadHex.match(/.{1,2}/g).map(b => parseInt(b, 16)));
        const bits = bytesToBits(payloadBytes);

        const width = image.width;
        const height = image.height;
        const channels = image.channels;
        const data = image.data;

        const maxBits = width * height * 3;
        if (bits.length > maxBits) {
            throw new Error(`Data too large! Need ${bits.length} bits, have ${maxBits}.`);
        }
        let bitIndex = 0;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                if (bitIndex >= bits.length) break;
                const idx = (y * width + x) * channels;
                for (let c = 0; c < 3 && c < channels; c++) {
                    if (bitIndex >= bits.length) break;
                    data[idx + c] = (data[idx + c] & 0xFE) | bits[bitIndex];
                    bitIndex++;
                }
            }
        }

        const outBuffer = await image.toBuffer({
            mimeType: 'image/png',
            pngCompressionLevel: 0
        });

        writeFileSync(outputPath, outBuffer);
        console.log(`OK: Embedded ${payloadBytes.length} bytes`);
        process.exit(0);

    } catch (error) {
        console.error(`ERROR: ${error.message}`);
        if(error.stack) console.error(error.stack);
        process.exit(1);
    }
}


async function extractData(imagePath, bitCount) {
    try {
        const buffer = readFileSync(imagePath);
        const image = await Image.load(buffer);

        const width = image.width;
        const height = image.height;
        const channels = image.channels;
        const data = image.data;
        const bits = [];

        for (let y = 0; y < height && bits.length < bitCount; y++) {
            for (let x = 0; x < width && bits.length < bitCount; x++) {
                const idx = (y * width + x) * channels;
                for (let c = 0; c < 3 && c < channels && bits.length < bitCount; c++) {
                    bits.push(data[idx + c] & 1);
                }
            }
        }

        const resultBytes = bitsToBytes(bits);
        console.log(Array.from(resultBytes).map(b => b.toString(16).padStart(2, '0')).join(''));
        process.exit(0);

    } catch (error) {
        console.error(`EXTRACT_ERROR: ${error.message}`);
        if(error.stack) console.error(error.stack);
        process.exit(1);
    }
}


const args = process.argv.slice(2);
const mode = args[0];

if (mode === 'embed') {
    if (args.length !== 3) {
        console.error("Usage: node worker.js embed <in> <out> <hex_payload>");
        process.exit(1);
    }
    const payloadHex = await readStdin();
    embedData(args[1], args[2], payloadHex);

} else if (mode === 'extract') {
    if (args.length !== 3) {
        console.error("Usage: node worker.js extract <image> <bit_count>");
        process.exit(1);
    }
    extractData(args[1], parseInt(args[2]));

} else {
    console.error("Unknown mode. Use 'embed' or 'extract'.");
    process.exit(1);
}