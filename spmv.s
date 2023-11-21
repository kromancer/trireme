	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0
	.p2align	2                               ; -- Begin function lexInsertF64
l_lexInsertF64:                         ; @lexInsertF64
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #80
	stp	x29, x30, [sp, #64]             ; 16-byte Folded Spill
	.cfi_def_cfa_offset 80
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	ldr	x8, [sp, #80]
	stp	x2, x3, [sp, #32]
	mov	x2, sp
	stp	x4, x5, [sp, #48]
	stp	x8, x1, [sp, #16]
	add	x1, sp, #24
	stp	x6, x7, [sp]
	bl	__mlir_ciface_lexInsertF64
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function newSparseTensor
l_newSparseTensor:                      ; @newSparseTensor
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #256
	stp	x28, x27, [sp, #224]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #240]            ; 16-byte Folded Spill
	.cfi_def_cfa_offset 256
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w27, -24
	.cfi_offset w28, -32
	ldp	x8, x9, [sp, #256]
	ldr	q0, [sp, #288]
	stp	x5, x6, [sp, #144]
	ldr	w5, [sp, #392]
	ldr	w6, [sp, #396]
	stur	q0, [sp, #120]
	ldr	x11, [sp, #408]
	stp	x7, x8, [sp, #160]
	ldr	w7, [sp, #400]
	stp	x9, x0, [sp, #176]
	add	x0, sp, #184
	ldp	x10, x8, [sp, #272]
	stp	x3, x4, [sp, #208]
	add	x3, sp, #64
	add	x4, sp, #24
	stp	x1, x2, [sp, #192]
	add	x1, sp, #144
	add	x2, sp, #104
	str	x11, [sp, #8]
	stp	x10, x8, [sp, #104]
	ldp	x9, x8, [sp, #304]
	ldr	x10, [sp, #320]
	str	x9, [sp, #136]
	add	x9, sp, #73
	stp	x8, x10, [sp, #64]
	ldp	x8, x10, [sp, #344]
	ldur	q0, [x9, #255]
	ldr	x9, [sp, #384]
	str	q0, [sp, #80]
	ldr	q0, [sp, #368]
	str	x8, [sp, #96]
	ldr	x8, [sp, #360]
	str	x9, [sp, #56]
	ldr	w9, [sp, #404]
	stur	q0, [sp, #40]
	stp	x10, x8, [sp, #24]
	str	w9, [sp]
	bl	__mlir_ciface_newSparseTensor
	ldp	x29, x30, [sp, #240]            ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #224]            ; 16-byte Folded Reload
	add	sp, sp, #256
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function sparseValuesF64
l_sparseValuesF64:                      ; @sparseValuesF64
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #64
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	.cfi_def_cfa_offset 64
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x1, x0
	add	x0, sp, #8
	bl	__mlir_ciface_sparseValuesF64
	ldp	x0, x1, [sp, #8]
	ldr	x4, [sp, #40]
	ldp	x2, x3, [sp, #24]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function sparseCoordinates0
l_sparseCoordinates0:                   ; @sparseCoordinates0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #64
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	.cfi_def_cfa_offset 64
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x2, x1
	mov	x1, x0
	add	x0, sp, #8
	bl	__mlir_ciface_sparseCoordinates0
	ldp	x0, x1, [sp, #8]
	ldr	x4, [sp, #40]
	ldp	x2, x3, [sp, #24]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function sparsePositions0
l_sparsePositions0:                     ; @sparsePositions0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #64
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	.cfi_def_cfa_offset 64
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x2, x1
	mov	x1, x0
	add	x0, sp, #8
	bl	__mlir_ciface_sparsePositions0
	ldp	x0, x1, [sp, #8]
	ldr	x4, [sp, #40]
	ldp	x2, x3, [sp, #24]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_spMV                           ; -- Begin function spMV
	.p2align	2
_spMV:                                  ; @spMV
	.cfi_startproc
; %bb.0:
	stp	x28, x27, [sp, #-96]!           ; 16-byte Folded Spill
	stp	x26, x25, [sp, #16]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #32]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #80]             ; 16-byte Folded Spill
	.cfi_def_cfa_offset 96
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	ldp	x22, x20, [sp, #104]
	mov	w1, #1                          ; =0x1
	ldr	x24, [sp, #96]
	mov	x19, x7
	mov	x21, x6
	mov	x23, x2
	mov	x27, x0
	bl	l_sparsePositions0
	mov	x25, x1
	mov	x0, x27
	mov	w1, #1                          ; =0x1
	bl	l_sparseCoordinates0
	mov	x0, x27
	mov	x26, x1
	bl	l_sparseValuesF64
	mov	x8, xzr
	b	LBB5_2
LBB5_1:                                 ;   in Loop: Header=BB5_2 Depth=1
	str	d0, [x19, x8, lsl #3]
	add	x8, x8, #1
LBB5_2:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_4 Depth 2
	cmp	x8, #2
	b.gt	LBB5_5
; %bb.3:                                ;   in Loop: Header=BB5_2 Depth=1
	lsl	x9, x8, #3
	add	x10, x25, x9
	ldr	d0, [x19, x9]
	ldp	x9, x10, [x10]
	cmp	x9, x10
	b.ge	LBB5_1
LBB5_4:                                 ;   Parent Loop BB5_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	lsl	x11, x9, #3
	add	x9, x9, #1
	ldr	x12, [x26, x11]
	ldr	d1, [x1, x11]
	ldr	d2, [x23, x12, lsl #3]
	fmul	d1, d1, d2
	fadd	d0, d0, d1
	cmp	x9, x10
	b.lt	LBB5_4
	b	LBB5_1
LBB5_5:
	mov	x0, x21
	mov	x1, x19
	mov	x2, x24
	mov	x3, x22
	mov	x4, x20
	ldp	x29, x30, [sp, #80]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #64]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #48]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #16]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp], #96             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #400
	stp	x28, x27, [sp, #304]            ; 16-byte Folded Spill
	stp	x26, x25, [sp, #320]            ; 16-byte Folded Spill
	stp	x24, x23, [sp, #336]            ; 16-byte Folded Spill
	stp	x22, x21, [sp, #352]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #368]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #384]            ; 16-byte Folded Spill
	.cfi_def_cfa_offset 400
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	ldr	x8, [sp, #424]
	ldr	x9, [sp, #416]
	mov	x24, x1
	ldr	x10, [sp, #400]
	mov	x2, xzr
	add	x0, sp, #280
	stp	x7, x8, [sp, #224]              ; 16-byte Folded Spill
	ldr	x8, [sp, #408]
	mov	x7, xzr
	add	x1, sp, #280
	add	x5, sp, #280
	add	x6, sp, #280
	stp	x8, x9, [sp, #208]              ; 16-byte Folded Spill
	ldp	x20, x9, [sp, #456]
	mov	w8, #2052                       ; =0x804
	mov	w3, #2                          ; =0x2
	mov	w4, #1                          ; =0x1
	strh	w8, [sp, #302]
	mov	w8, #3                          ; =0x3
	stp	x9, x10, [sp, #192]             ; 16-byte Folded Spill
	mov	w9, #4                          ; =0x4
	add	x10, sp, #264
	stp	x8, x9, [sp, #280]
	ldp	x9, x19, [sp, #440]
	ldr	x8, [sp, #432]
	stp	x10, x10, [sp, #96]
	stp	x10, xzr, [sp, #64]
	stp	x8, x9, [sp, #176]              ; 16-byte Folded Spill
	mov	w8, #1                          ; =0x1
	mov	w9, #2                          ; =0x2
	stp	x8, x10, [sp, #48]
	add	x10, sp, #302
	stp	xzr, x8, [sp, #264]
	stp	x8, xzr, [sp, #144]
	stp	x8, xzr, [sp, #128]
	stp	xzr, x9, [sp, #112]
	stp	x9, x8, [sp, #80]
	stp	x10, x10, [sp, #16]
	stp	x9, x8, [sp]
	stp	xzr, x9, [sp, #32]
	bl	l_newSparseTensor
	mov	x27, x0
	mov	x21, xzr
	mov	x22, xzr
	add	x23, sp, #248
	add	x28, sp, #240
	b	LBB6_2
LBB6_1:                                 ;   in Loop: Header=BB6_2 Depth=1
	add	x22, x22, #1
	add	x21, x21, #32
LBB6_2:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_5 Depth 2
	cmp	x22, #2
	b.gt	LBB6_8
; %bb.3:                                ; %.preheader
                                        ;   in Loop: Header=BB6_2 Depth=1
	mov	x25, xzr
	add	x26, x24, x21
	b	LBB6_5
LBB6_4:                                 ;   in Loop: Header=BB6_5 Depth=2
	add	x25, x25, #1
LBB6_5:                                 ;   Parent Loop BB6_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	cmp	x25, #3
	b.gt	LBB6_1
; %bb.6:                                ;   in Loop: Header=BB6_5 Depth=2
	ldr	d0, [x26, x25, lsl #3]
	fcmp	d0, #0.0
	b.eq	LBB6_4
; %bb.7:                                ;   in Loop: Header=BB6_5 Depth=2
	add	x1, sp, #248
	add	x2, sp, #248
	add	x6, sp, #240
	add	x7, sp, #240
	mov	x0, x27
	mov	x3, xzr
	mov	w4, #2                          ; =0x2
	mov	w5, #1                          ; =0x1
	stp	x22, x25, [x23]
	str	d0, [x28]
	str	xzr, [sp]
	bl	l_lexInsertF64
	b	LBB6_4
LBB6_8:
	mov	x0, x27
	bl	_endLexInsert
	ldp	x4, x1, [sp, #216]              ; 16-byte Folded Reload
	mov	x0, x27
	ldp	x2, x3, [sp, #200]              ; 16-byte Folded Reload
	ldr	x5, [sp, #232]                  ; 8-byte Folded Reload
	ldp	x6, x7, [sp, #176]              ; 16-byte Folded Reload
	ldr	x8, [sp, #192]                  ; 8-byte Folded Reload
	stp	x19, x20, [sp]
	str	x8, [sp, #16]
	bl	_spMV
	ldp	x29, x30, [sp, #384]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #368]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #352]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #336]            ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #320]            ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #304]            ; 16-byte Folded Reload
	add	sp, sp, #400
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__mlir_ciface_main              ; -- Begin function _mlir_ciface_main
	.p2align	2
__mlir_ciface_main:                     ; @_mlir_ciface_main
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #112
	stp	x20, x19, [sp, #80]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #96]             ; 16-byte Folded Spill
	.cfi_def_cfa_offset 112
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	ldp	x8, x4, [x1, #24]
	mov	x19, x0
	ldp	x10, x9, [x1, #8]
	ldr	x11, [x1]
	ldp	x5, x6, [x1, #40]
	ldr	x7, [x2]
	mov	x0, x11
	ldur	q0, [x2, #8]
	ldr	x14, [x3, #32]
	mov	x1, x10
	ldp	x12, x13, [x2, #24]
	mov	x2, x9
	ldp	q1, q2, [x3]
	mov	x3, x8
	str	x14, [sp, #64]
	stp	x12, x13, [sp, #16]
	stp	q1, q2, [sp, #32]
	str	q0, [sp]
	bl	_main
	stp	x0, x1, [x19]
	ldp	x29, x30, [sp, #96]             ; 16-byte Folded Reload
	stp	x2, x3, [x19, #16]
	str	x4, [x19, #32]
	ldp	x20, x19, [sp, #80]             ; 16-byte Folded Reload
	add	sp, sp, #112
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
