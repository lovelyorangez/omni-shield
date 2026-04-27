// This file is MIT Licensed.
//
// Copyright 2017 Christian Reitwiessner
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
pragma solidity ^0.8.0;
library Pairing {
    struct G1Point {
        uint X;
        uint Y;
    }
    // Encoding of field elements is: X[0] * z + X[1]
    struct G2Point {
        uint[2] X;
        uint[2] Y;
    }
    /// @return the generator of G1
    function P1() pure internal returns (G1Point memory) {
        return G1Point(1, 2);
    }
    /// @return the generator of G2
    function P2() pure internal returns (G2Point memory) {
        return G2Point(
            [10857046999023057135944570762232829481370756359578518086990519993285655852781,
             11559732032986387107991004021392285783925812861821192530917403151452391805634],
            [8495653923123431417604973247489272438418190587263600148770280649306958101930,
             4082367875863433681332203403145435568316851327593401208105741076214120093531]
        );
    }
    /// @return the negation of p, i.e. p.addition(p.negate()) should be zero.
    function negate(G1Point memory p) pure internal returns (G1Point memory) {
        // The prime q in the base field F_q for G1
        uint q = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
        if (p.X == 0 && p.Y == 0)
            return G1Point(0, 0);
        return G1Point(p.X, q - (p.Y % q));
    }
    /// @return r the sum of two points of G1
    function addition(G1Point memory p1, G1Point memory p2) internal view returns (G1Point memory r) {
        uint[4] memory input;
        input[0] = p1.X;
        input[1] = p1.Y;
        input[2] = p2.X;
        input[3] = p2.Y;
        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 6, input, 0xc0, r, 0x60)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }
        require(success);
    }


    /// @return r the product of a point on G1 and a scalar, i.e.
    /// p == p.scalar_mul(1) and p.addition(p) == p.scalar_mul(2) for all points p.
    function scalar_mul(G1Point memory p, uint s) internal view returns (G1Point memory r) {
        uint[3] memory input;
        input[0] = p.X;
        input[1] = p.Y;
        input[2] = s;
        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 7, input, 0x80, r, 0x60)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }
        require (success);
    }
    /// @return the result of computing the pairing check
    /// e(p1[0], p2[0]) *  .... * e(p1[n], p2[n]) == 1
    /// For example pairing([P1(), P1().negate()], [P2(), P2()]) should
    /// return true.
    function pairing(G1Point[] memory p1, G2Point[] memory p2) internal view returns (bool) {
        require(p1.length == p2.length);
        uint elements = p1.length;
        uint inputSize = elements * 6;
        uint[] memory input = new uint[](inputSize);
        for (uint i = 0; i < elements; i++)
        {
            input[i * 6 + 0] = p1[i].X;
            input[i * 6 + 1] = p1[i].Y;
            input[i * 6 + 2] = p2[i].X[1];
            input[i * 6 + 3] = p2[i].X[0];
            input[i * 6 + 4] = p2[i].Y[1];
            input[i * 6 + 5] = p2[i].Y[0];
        }
        uint[1] memory out;
        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 8, add(input, 0x20), mul(inputSize, 0x20), out, 0x20)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }
        require(success);
        return out[0] != 0;
    }
    /// Convenience method for a pairing check for two pairs.
    function pairingProd2(G1Point memory a1, G2Point memory a2, G1Point memory b1, G2Point memory b2) internal view returns (bool) {
        G1Point[] memory p1 = new G1Point[](2);
        G2Point[] memory p2 = new G2Point[](2);
        p1[0] = a1;
        p1[1] = b1;
        p2[0] = a2;
        p2[1] = b2;
        return pairing(p1, p2);
    }
    /// Convenience method for a pairing check for three pairs.
    function pairingProd3(
            G1Point memory a1, G2Point memory a2,
            G1Point memory b1, G2Point memory b2,
            G1Point memory c1, G2Point memory c2
    ) internal view returns (bool) {
        G1Point[] memory p1 = new G1Point[](3);
        G2Point[] memory p2 = new G2Point[](3);
        p1[0] = a1;
        p1[1] = b1;
        p1[2] = c1;
        p2[0] = a2;
        p2[1] = b2;
        p2[2] = c2;
        return pairing(p1, p2);
    }
    /// Convenience method for a pairing check for four pairs.
    function pairingProd4(
            G1Point memory a1, G2Point memory a2,
            G1Point memory b1, G2Point memory b2,
            G1Point memory c1, G2Point memory c2,
            G1Point memory d1, G2Point memory d2
    ) internal view returns (bool) {
        G1Point[] memory p1 = new G1Point[](4);
        G2Point[] memory p2 = new G2Point[](4);
        p1[0] = a1;
        p1[1] = b1;
        p1[2] = c1;
        p1[3] = d1;
        p2[0] = a2;
        p2[1] = b2;
        p2[2] = c2;
        p2[3] = d2;
        return pairing(p1, p2);
    }
}

contract Verifier {
    using Pairing for *;
    struct VerifyingKey {
        Pairing.G1Point alpha;
        Pairing.G2Point beta;
        Pairing.G2Point gamma;
        Pairing.G2Point delta;
        Pairing.G1Point[] gamma_abc;
    }
    struct Proof {
        Pairing.G1Point a;
        Pairing.G2Point b;
        Pairing.G1Point c;
    }
    function verifyingKey() pure internal returns (VerifyingKey memory vk) {
        vk.alpha = Pairing.G1Point(uint256(0x1d229c1d447aeb09331433867e7508ebeafba6dc6003b3c10b150aafb671e9d7), uint256(0x15c1615978f3f1e36bc14f32c5c45611b09d5cacc13ddaccec184e7ca1694b49));
        vk.beta = Pairing.G2Point([uint256(0x12d4bccce08733d3c373a38df5f005fb3b39e2883385e56e2b0b8e9e74c0a25a), uint256(0x005c13113128c6da33140410acf05f5490fff492a7699b1e78d69d2f80477183)], [uint256(0x27fd1838485296c90385c18ae0ea7215718bceec2a392cfd90797994d7f5816c), uint256(0x23d13d4084a9477fe1fac59299abd6fcbdd6a454c71da9b7ba4257c01edbd523)]);
        vk.gamma = Pairing.G2Point([uint256(0x0507fd18859a970b0eedd303db20212325b309b74687183977d36e117241bf8e), uint256(0x29851b887a6f7f87ce6e0c2c4504c2dc05cfbeebe73c02d55ac614e39c7a50cc)], [uint256(0x26db714836b063ecca551f3bcfd715466cb0df31c28ae49b5161600b82eec8a1), uint256(0x0833e3311785635b10217ccc61a91996c20b5f8e0d115ba6eb8e10fdb0522b1d)]);
        vk.delta = Pairing.G2Point([uint256(0x22e9241325f3f1dd783c5eb3b36e24cf6f83f555299e581e15feddc986f8c265), uint256(0x10df4700deb40bfc5d95d19ab1ebc804445e0917d3f08fde50613867a8e4ccb0)], [uint256(0x160cd3841d2c1e66610d7c69e98977859f6c1f4729965cc0aec9e4ac58122b2f), uint256(0x01ab114cd13618ecfdf67510ee9099b8eeb3e760c7b60c1e4185d7f2ba8b64a6)]);
        vk.gamma_abc = new Pairing.G1Point[](18);
        vk.gamma_abc[0] = Pairing.G1Point(uint256(0x0ecb8eee6fcf50ecdfbd07edd53bea879a24beb2231dc59bf01dd8d70c8faf7b), uint256(0x058bff08e10b9566df10b4a970f9e428bb74cd061bc50704d117fa406fb6c650));
        vk.gamma_abc[1] = Pairing.G1Point(uint256(0x23e99b70b6fcc4994a73a8c62ea9b951ef387bfe60e23316b6b7cc4b204544d3), uint256(0x168f38e8499d6f25e9028eb406b520ddad781d442970f764330d46a8dea8f3f4));
        vk.gamma_abc[2] = Pairing.G1Point(uint256(0x0f1eec84b859a06823907e05d39246ae91c83d5b3ad4e01f49d16c7ed5f0dde1), uint256(0x2857a2bef091488a22cdbbd72f927713b1e9467f61048685525ede78e7df35e9));
        vk.gamma_abc[3] = Pairing.G1Point(uint256(0x09f520587ad87705b32018079aa3e667bd69abfd66151b8dddabb5fce28ecd9f), uint256(0x1b969ec9c505700c7c674eb73c91c4714fa5e2e9fb872e30a2cc4fdb546ae48d));
        vk.gamma_abc[4] = Pairing.G1Point(uint256(0x1fa2fcd2829e6e040109673d5a90b29fe15865d55d9835d304944e4a92d1a433), uint256(0x226ab56ee34283d4ee4e5082c9fcb38209ef6ff494f1b530ec72174460fcda8e));
        vk.gamma_abc[5] = Pairing.G1Point(uint256(0x2a68c7d63aec7158281c103328403b47d3694fa687df6a3f5201f0595947e6a4), uint256(0x05a5b2c990d03c77e01889c60e2d7ec3796e4f3a119ffc0935b5809d9dd56f9e));
        vk.gamma_abc[6] = Pairing.G1Point(uint256(0x1b0c39576508b9404d195110d691ca988a81863c65ffe5ecfbae619037936920), uint256(0x2061ad0ede45f6d1d0e0cc16e431dab070f7b321f06e5f7fd9e39455dab947ea));
        vk.gamma_abc[7] = Pairing.G1Point(uint256(0x095982f9a885bb30394af1f5bf5333f9b3e1f0b5173af2af9fb6e253d6ec8b6c), uint256(0x0c41f02a02b586d9784438338c703ed765897328bf0011ebfacb1363e3a5cbcf));
        vk.gamma_abc[8] = Pairing.G1Point(uint256(0x025b5dc161288730da7cc1b2548790b993908f514e02eca09de53531754a21ca), uint256(0x15ce17db473bf90ac69e663544271e4b0c7f3dc5966b6557c9e253d0f978ef2f));
        vk.gamma_abc[9] = Pairing.G1Point(uint256(0x2b0befdd68a8ac59611af85d523eb86cc2bce0cda24790183ef2ccfcb116506e), uint256(0x1954b3acf8e1e8729790f3177892f570bab44d3ca45bb9bc91ae9359e6ab7695));
        vk.gamma_abc[10] = Pairing.G1Point(uint256(0x127d23bdb69bb47056375f9e95ced1e945309eb367dc5482e8487be9056b39f8), uint256(0x2451ef192176c09a2b82a26b1035da5ee4c596323db9b428e2573ce0c572a2a3));
        vk.gamma_abc[11] = Pairing.G1Point(uint256(0x19371f1deb259f747637495b9fbc5aaf9a20318bd546e0be2237b65c47e11f02), uint256(0x01c8a7b87896511aed0aacbfbf0018516f2e3deb4934bc1a6ce0095b562b9b94));
        vk.gamma_abc[12] = Pairing.G1Point(uint256(0x28c80aec6a31283fd6aa4b46d52c6ce86979bfbcfe476ec28bac29a5ebd45814), uint256(0x2143510c24f551c742d30f6e98802a0dee3f416c9b318786e01c9f3af0928842));
        vk.gamma_abc[13] = Pairing.G1Point(uint256(0x22731f269bb0334945232b8c0e192e79c68188f45315e512f6f5fef8aa2130a7), uint256(0x1a124f371c1470dbf12848684c37fcf65a5426eac8d596607670762b089f5ef2));
        vk.gamma_abc[14] = Pairing.G1Point(uint256(0x0e5f57324503bb06f2ff525fc1e90de46f3368279a55c3670c3b04ac9752e9f8), uint256(0x2067e63686811f3d832d593973a37cdf933b0ee6d6f622db811bac9d4bdf6b7f));
        vk.gamma_abc[15] = Pairing.G1Point(uint256(0x1fc89e7b429f1972d78f1ab7d3fe9ae019911a3ec2201f3dbdd1793dd69573d4), uint256(0x079a511a6e76ad0fbcde1dbb551c5fb9b5ee1fa6072cecdb617c5f41a6194a01));
        vk.gamma_abc[16] = Pairing.G1Point(uint256(0x2b24709f639f23c46dcdc5d308e701a1b175fb6177db43a2ebd7d460ca78bfc6), uint256(0x019f0b555591dc99521c45d916a8cb2649a0bc6abc311f12b90835220bc485a6));
        vk.gamma_abc[17] = Pairing.G1Point(uint256(0x0fa85852871d93c2be0bd14578ce4ad5805f5a8cb503b6852c66742a92879f4d), uint256(0x0a17a55212e74059650eef2c9a5f9aeec5fbf846b77fce937c92c51146b4a8ba));
    }
    function verify(uint[] memory input, Proof memory proof) internal view returns (uint) {
        uint256 snark_scalar_field = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
        VerifyingKey memory vk = verifyingKey();
        require(input.length + 1 == vk.gamma_abc.length);
        // Compute the linear combination vk_x
        Pairing.G1Point memory vk_x = Pairing.G1Point(0, 0);
        for (uint i = 0; i < input.length; i++) {
            require(input[i] < snark_scalar_field);
            vk_x = Pairing.addition(vk_x, Pairing.scalar_mul(vk.gamma_abc[i + 1], input[i]));
        }
        vk_x = Pairing.addition(vk_x, vk.gamma_abc[0]);
        if(!Pairing.pairingProd4(
             proof.a, proof.b,
             Pairing.negate(vk_x), vk.gamma,
             Pairing.negate(proof.c), vk.delta,
             Pairing.negate(vk.alpha), vk.beta)) return 1;
        return 0;
    }
    function verifyTx(
            Proof memory proof, uint[17] memory input
        ) public view returns (bool r) {
        uint[] memory inputValues = new uint[](17);
        
        for(uint i = 0; i < input.length; i++){
            inputValues[i] = input[i];
        }
        if (verify(inputValues, proof) == 0) {
            return true;
        } else {
            return false;
        }
    }
}
